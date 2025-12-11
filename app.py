import os
import sys
import warnings

# Suppress Warnings (PIL & PyTorch non-critical)
warnings.filterwarnings("ignore")

# Force UTF-8 for Windows Console to prevent 'latin-1' crashes
try:
    sys.stdout.reconfigure(encoding='utf-8')
except: pass

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import io
import re
import zipfile
import spacy
import phonenumbers
import easyocr
import openpyxl
from dotenv import load_dotenv
import cv2
import json
import traceback
import torch
from streamlit_lottie import st_lottie
import time
import concurrent.futures
import uuid
import hashlib
import extra_streamlit_components as stx # Imported extra_streamlit_components

# GenAI Imports
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tldextract import tldextract

# Page Configuration
st.set_page_config(page_title="Bulk Business Card OCR → Excel", layout="wide")

# --- Helper Functions & Caching ---

@st.cache_resource
def load_spacy_model():
    """Load spaCy model."""
    try:
        # Model is installed via requirements.txt
        return spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Spacy Error: {e}")
        return None

@st.cache_resource
def load_ocr_model(use_gpu=True):
    """Load EasyOCR with robust Fallback."""
    try:
        if use_gpu and torch.cuda.is_available():
            try:
                # Test Allocation to catch CUBLAS Error Early
                t = torch.tensor([1.0]).cuda()
                del t
                torch.cuda.empty_cache()
                return easyocr.Reader(['en'], gpu=True)
            except Exception as e:
                print(f"GPU Init Failed (Switched to CPU): {e}")
                return easyocr.Reader(['en'], gpu=False)
        else:
            return easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        st.error(f"OCR Model Error: {e}")
        return None

def preprocess_image_bytes(image_bytes):
    """
    Advanced Preprocessing for OCR (1080p Optimization):
    1. Read Image & Convert to BGR
    2. Resize to fixed width 1080px (Upscale/Downscale)
    3. Grayscale
    4. Denoise (Gaussian)
    5. Histogram Equalization (CLAHE)
    6. Unsharp Masking (Sharpening)
    """
    image = PIL.Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB': image = image.convert('RGB')
    img_array = np.array(image)

    # 1. Standardize Color Space
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # 2. Resolution Normalization (Target Width: 1080px)
    target_width = 1080
    height, width, _ = img_bgr.shape
    scale = target_width / width

    if scale != 1.0:
        if scale > 1.0:
            # Upscale (Cubic is best for enlarging)
            img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            # Downscale (Area is best for shrinking)
            img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 3. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 4. Mild Denoising (to reduce noise from sharpening later)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # 5. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrasted = clahe.apply(denoised)

    # 6. Sharpening (Unsharp Mask)
    # Create specific kernel for text edge enhancement
    sharpen_kernel = np.array([[0, -1, 0], 
                               [-1, 5,-1], 
                               [0, -1, 0]])
    sharpened = cv2.filter2D(contrasted, -1, sharpen_kernel)

    return sharpened

def load_lottieurl(url: str):
    import requests
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# --- Parsing Logic ---

from phonenumbers import PhoneNumberMatcher

def parse_extracted_text_heuristic(text_lines, nlp_model):
    """Legacy Regex/Heuristic Parser."""
    text_lines = [line.strip() for line in text_lines if len(line.strip()) > 1]
    raw_text = "\n".join(text_lines)
    # Fallback if spacy failed
    doc = nlp_model(raw_text) if nlp_model else None

    data = {"Name": None, "Designation": None, "Company": None, "Phone1": None, "Phone2": None, 
            "Email1": None, "Email2": None, "Website": None, "Address": None, "Raw_Text": raw_text}

    # Emails
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = list(dict.fromkeys(re.findall(email_pattern, raw_text, re.IGNORECASE)))
    if len(emails) > 0: data["Email1"] = emails[0]
    if len(emails) > 1: data["Email2"] = emails[1]

    # Phones
    phones = []
    try:
        for match in PhoneNumberMatcher(raw_text, "IN"): 
            phones.append(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.E164))
    except: pass
    if not phones:
        potential_phones = re.findall(r'(?:\+?\d{1,3}[ -]?)?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}', raw_text)
        phones.extend([p for p in potential_phones if len(re.sub(r'\D', '', p)) >= 10])
    phones = list(dict.fromkeys(phones))
    if len(phones) > 0: data["Phone1"] = phones[0]
    if len(phones) > 1: data["Phone2"] = phones[1]

    # Website
    url_pattern = r'(https?://[^\s]+)|(www\.[^\s]+)|([a-zA-Z0-9][a-zA-Z0-9-]+\.(com|org|net|gov|edu|io|co|in|ai|biz|info)[^\s]*)'
    potential_urls = []
    for line in text_lines:
        if "@" in line: continue 
        match = re.search(url_pattern, line, re.IGNORECASE)
        if match:
             cand = match.group(0).rstrip(',.')
             if "www." in cand.lower() or "http" in cand.lower(): potential_urls.append(cand)
             else:
                 try:
                    ext = tldextract.extract(cand)
                    if ext.domain and ext.suffix: potential_urls.append(cand)
                 except: pass
    if potential_urls: data["Website"] = potential_urls[0]

    # Designation
    designation_keywords = ["Manager", "Senior", "Engineer", "Developer", "Director", "Founder", "Co-Founder", "CEO", "CTO", "CFO", "COO", "Consultant", "Head", "Lead", "Associate", "President", "VP", "Partner", "Executive", "Officer", "Specialist", "Analyst", "Sales"]
    desig_line_idx = -1
    for i, line in enumerate(text_lines):
        if any(k.lower() in line.lower() for k in designation_keywords):
            data["Designation"] = line
            desig_line_idx = i
            break

    # Name
    def is_noise(line):
        return "@" in line or "www." in line.lower() or any(char.isdigit() for char in line) or len(line.split()) > 5
    found_name = False

    if doc:
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not is_noise(ent.text):
                data["Name"] = ent.text; found_name = True; break

    if not found_name:
        for i, line in enumerate(text_lines[:3]):
            if i == desig_line_idx: continue
            clean = line.strip()
            if len(clean) > 2 and not is_noise(clean):
                corp_kw = ["Pvt", "Ltd", "Inc", "Corp", "LLC"]
                if not any(k.lower() in clean.lower() for k in corp_kw):
                    data["Name"] = clean; break

    # Company
    company_keywords = ["Pvt Ltd", "Private Limited", "LLP", "Ltd", "Inc", "Corp", "Corporation", "LLC", "GmbH", "Group", "Solutions", "Services"]
    found_company = False
    for i, line in enumerate(text_lines):
        if i == desig_line_idx: continue
        if any(k.lower() in line.lower() for k in company_keywords):
            data["Company"] = line; found_company = True; break
    if not found_company and doc:
        for ent in doc.ents:
            if ent.label_ == "ORG" and len(ent.text) > 3 and ent.text != data["Name"]:
                 data["Company"] = ent.text; break

    # Address
    address_parts = []
    addr_keys = ["Road", "Rd", "Street", "St", "Marg", "Nagar", "Ave", "Floor", "Block", "Tower", "Plot", "City", "State", "Pin", "Zip", "India", "USA", "UK", "Box", "Room"]
    for line in text_lines:
        if line in [data["Name"], data["Company"], data["Designation"]]: continue
        if data["Email1"] and line in data["Email1"]: continue
        if data["Website"] and line in data["Website"]: continue
        if bool(re.search(r'\b\d{5,6}\b', line)) or any(k.lower() in re.split(r'[ ,]', line.lower()) for k in addr_keys):
             address_parts.append(line)
    if address_parts: data["Address"] = ", ".join(address_parts)

    data["Status"] = "OK" if (data["Name"] and (data["Phone1"] or data["Email1"])) else "CHECK"
    return data

# --- GenAI Pipeline Logic ---

def correct_ocr_with_deepseek(raw_text, hf_token):
    try:
        repo_id = "meta-llama/Llama-3.1-8B-Instruct" 
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.1,
            huggingfacehub_api_token=hf_token,
            timeout=120
        )
        chat_model = ChatHuggingFace(llm=llm)

        system_prompt = """You are an expert OCR Post-Processing AI. 
1. Fix basic OCR errors ('0' vs 'O', 'l' vs '1').
2. CRITICAL: Reformat Jumbled Layouts. If multiple fields appear on one line (e.g., "Name Phone Email"), SEPARATE them into distinct lines.
3. Fix broken URLs/Emails (e.g., "mail @ domain . com" -> "mail@domain.com").
Return ONLY the corrected, clean text."""

        # Use literal string with {raw_text} placeholder for LangChain
        user_template = "Correct this:\n{raw_text}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])

        chain = prompt | chat_model
        response = chain.invoke({"raw_text": raw_text})
        return response.content

    except Exception as e:
        # Safe Print to avoid crash
        try: print(f"Correction failed: {e}")
        except: pass
        return raw_text

def extract_genai_pure(corrected_text, hf_token):
    try:
        repo_id = "meta-llama/Llama-3.1-8B-Instruct"
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.1,
            huggingfacehub_api_token=hf_token,
            timeout=120
        )
        chat_model = ChatHuggingFace(llm=llm)

        system_prompt = """You are a master Business Card Intelligence Agent.
Your Goal: Extract structured data from messy, jumbled, or horizontal OCR text.

STRICT Extraction Rules:
1. **Un-Jumble Text:** The OCR often reads horizontally across columns. Re-assemble logical fields.
   - Example Raw: "Vedant 12345 Company"
   - Interpretation: "Name: Vedant", "Phone: 12345", "Company: Company".
   
2. **Mandatory Extraction (No Omissions):**
   - **Email:** If you see '@' or 'gmail/outlook', you MUST extract it. Join broken parts (e.g. "x @ y. z" -> "x@y.z").
   - **Website:** If you see 'www', '.com', '.in', 'http', you MUST extract it.
   - **Phone:** Extract ALL numbers that look like phones.

3. **Disambiguation:**
   - **Company vs Name:** 
     - Companies often have: Pvt, Ltd, Inc, LLC, Solutions, Technologies, Group.
     - Names often are near Designations (Manager, CEO).
     - If unsure, the "biggest" bold-looking word is usually the Company, the standard noun is the Name.

4. **Output Format:**
   Return strict JSON with fields: Name, Designation, Company, Address, Phone (list), Email (list), Website (list).
   If a field is missing in text, interpret context to find it. Do not leave Name/Company empty unless absolutely impossible."""

        # Use literal string with {corrected_text} placeholder for LangChain
        user_msg = """Analyze this OCR Text carefully:

{corrected_text}

Return strictly VALID JSON. No Markdown. No Explanations."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_msg)
        ])

        chain = prompt | chat_model | JsonOutputParser()
        ai_data = chain.invoke({"corrected_text": corrected_text})

    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "Unauthorized" in err_msg:
             return {"error": "INVALID_KEY: Your Hugging Face API Token is incorrect. Please check it."}
        elif "429" in err_msg or "Rate limit" in err_msg:
             return {"error": "LIMIT_EXCEEDED: API Rate Limit Reached. Please create a NEW account & key on HuggingFace."}

        try: print(f"Extraction Error: {e}")
        except: pass
        ai_data = {}

    # Propagate Error
    if "error" in ai_data: return ai_data

    # Post-Process
    final_data = {
        "Name": ai_data.get("Name"),
        "Designation": ai_data.get("Designation"),
        "Company": ai_data.get("Company"),
        "Address": ai_data.get("Address"),
        "Phone": [], "Email": [], "Website": []
    }

    # Email
    ai_emails = ai_data.get("Email", [])
    if isinstance(ai_emails, str): ai_emails = [ai_emails]
    raw_emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', corrected_text)
    final_data["Email"] = list(set((ai_emails or []) + raw_emails))

    # Phone
    ai_phones = ai_data.get("Phone", [])
    if isinstance(ai_phones, str): ai_phones = [ai_phones]
    valid_phones = []
    for p in (ai_phones or []):
        if len(re.sub(r'\D', '', str(p))) > 5: valid_phones.append(str(p))
    if not valid_phones:
        valid_phones = re.findall(r'(\+?\d[\d -]{8,15}\d)', corrected_text)
    final_data["Phone"] = list(set(valid_phones))

    # Website
    ai_web = ai_data.get("Website", [])
    if isinstance(ai_web, str): ai_web = [ai_web]
    final_data["Website"] = ai_web or []

    # Flatten
    normalized = {
        "Name": final_data["Name"],
        "Designation": final_data["Designation"],
        "Company": final_data["Company"],
        "Phone1": final_data["Phone"][0] if len(final_data["Phone"]) > 0 else None,
        "Phone2": final_data["Phone"][1] if len(final_data["Phone"]) > 1 else None,
        "Email1": final_data["Email"][0] if len(final_data["Email"]) > 0 else None,
        "Email2": final_data["Email"][1] if len(final_data["Email"]) > 1 else None,
        "Website": final_data["Website"][0] if len(final_data["Website"]) > 0 else None,
        "Address": final_data["Address"],
        "Status": "GenAI_OK",
        "Raw_Text": corrected_text,
        "Model_Used": "Llama-3.1-8B"
    }
    return normalized

def extract_with_llm(text_lines, hf_token, nlp_model=None):
    raw_text = "\n".join(text_lines)
    corrected_text = correct_ocr_with_deepseek(raw_text, hf_token)

    # 1. GenAI Extraction
    data = extract_genai_pure(corrected_text, hf_token)

    # 2. Hybrid Fallback (Safety Net)
    # If GenAI failed to find critical info (Phone or Email) but Regex might find it.
    if nlp_model and "error" not in data:
        # Check for missing critical fields
        missing_phone = not data.get("Phone1")
        missing_email = not data.get("Email1")

        if missing_phone or missing_email:
            # Run "Dumb" Heuristic Parser on the SAME text
            print(f"GenAI missed data for record. Running Hybrid Fallback...")
            heuristic_data = parse_extracted_text_heuristic(text_lines, nlp_model)

            # Merge Strategy: Trust GenAI > Trust Regex, but fill GenAI gaps with Regex
            if missing_phone and heuristic_data.get("Phone1"):
                data["Phone1"] = heuristic_data["Phone1"]
                data["Phone2"] = heuristic_data.get("Phone2")
                data["Status"] += "_HybridPhone"

            if missing_email and heuristic_data.get("Email1"):
                data["Email1"] = heuristic_data["Email1"]
                data["Email2"] = heuristic_data.get("Email2")
                data["Status"] += "_HybridEmail"

    return data

# --- UI Styling ---
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3, p, div, span, label { color: #ffffff !important; }
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
    [data-testid='stFileUploader'] { width: 100%; padding: 30px; border: 2px dashed #4facfe; background-color: #1a1a1a; }
    .stButton>button { width: 100%; height: 50px; background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%); color: #000000; font-weight: 800; border: none; }
    div[data-testid="stDialog"] { background-color: #111; color: #fff; }
    /* Smaller button style for "Clear RAM" */
    .stButton>button.small-button { height: 35px !important; font-size: 0.8em; }
    /* Reduce Top Padding for Main App - increased to 6rem to push content down significantly */
    /* Reduce Top Padding for Main App - increased to 3.5rem */
    .block-container { padding-top: 3.5rem !important; }
    
    /* Compact Sidebar - Aggressive */
    section[data-testid="stSidebar"] > div > div:nth-of-type(2) { padding-top: 0rem !important; }
    /* Reduce gap between sidebar elements */
    section[data-testid="stSidebar"] .stElementContainer { margin-bottom: 0rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem !important; }
    
    /* Login Dialog Styling */
    div[data-testid="stDialog"] > div[role="dialog"] { 
        background-color: #1E1E1E !important; 
        border: 1px solid #333; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Hide Dialog Close Button */
    div[data-testid="stDialog"] button[aria-label="Close"] { display: none !important; }
    
    /* API Key Input Visibility */
    input[type="password"] {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Cookie Manager for Persistence ---
def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

# --- Authentication Layer ---

USERS_DB = "users_db.json"
DEFAULT_AVATAR = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"

def load_users():
    if not os.path.exists(USERS_DB): return {}
    try:
        with open(USERS_DB, "r") as f: return json.load(f)
    except: return {}

def save_user(username, password, avatar=DEFAULT_AVATAR):
    users = load_users()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    # Schema Standard: {"password": "...", "avatar": "..."}
    users[username] = {"password": hashed, "avatar": avatar}
    with open(USERS_DB, "w") as f: json.dump(users, f)

def update_user_avatar(username, avatar_url):
    users = load_users()
    if username in users:
        curr = users[username]
        # Handle Legacy (String) vs New (Dict)
        if isinstance(curr, str): curr = {"password": curr, "avatar": DEFAULT_AVATAR}
        curr["avatar"] = avatar_url
        users[username] = curr
        with open(USERS_DB, "w") as f: json.dump(users, f)

def verify_user(username, password):
    users = load_users()
    if username not in users: return False

    stored = users[username]
    # Handle Legacy
    if isinstance(stored, str): stored_hash = stored
    else: stored_hash = stored.get("password")

    hashed = hashlib.sha256(password.encode()).hexdigest()
    return stored_hash == hashed

def get_user_avatar(username):
    users = load_users()
    if username not in users: return DEFAULT_AVATAR
    stored = users[username]
    if isinstance(stored, str): return DEFAULT_AVATAR
    return stored.get("avatar", DEFAULT_AVATAR)

if "user_mode" not in st.session_state:
    st.session_state.user_mode = None 
    st.session_state.user_username = None

# Check Cookies for Auto-Login
time.sleep(0.1) # Wait for cookie manager
auth_cookie = cookie_manager.get("c_user")
if auth_cookie and st.session_state.user_mode is None:
    # Validate cookie (simple existence check for now, ideally verify hash)
    if auth_cookie == "guest":
        st.session_state.user_mode = "guest"
        st.session_state.user_username = "Guest"
    else:
        # Check if user exists
        users = load_users()
        if auth_cookie in users:
            st.session_state.user_mode = "user"
            st.session_state.user_username = auth_cookie

@st.dialog("Welcome to AI Business Card Extractor")
def login_dialog():
    st.subheader("Access Your Workspace")

    tab1, tab2, tab3 = st.tabs(["🔐 Login", "📝 Sign Up", "🚀 Guest"])

    with tab1:
        username = st.text_input("Username", key="l_user")
        password = st.text_input("Password", type="password", key="l_pass")
        if st.button("Log In"):
            if verify_user(username, password):
                st.session_state.user_mode = "user"
                st.session_state.user_username = username
                # Set Cookie (Expires in 5 days)
                cookie_manager.set("c_user", username, expires_at=pd.Timestamp.now() + pd.Timedelta(days=5))
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        s_user = st.text_input("Username", key="s_user")
        s_pass1 = st.text_input("Password", type="password", key="s_p1")
        s_pass2 = st.text_input("Confirm Password", type="password", key="s_p2")
        if st.button("Create Account"):
            if s_pass1 != s_pass2:
                st.error("Passwords do not match")
            elif len(s_pass1) < 4:
                st.error("Password must be at least 4 characters")
            elif not s_user:
                st.error("Username cannot be empty")
            elif load_users().get(s_user):
                st.error("Username already exists! Try a new one.")
            else:
                save_user(s_user, s_pass1)
                st.success("Account Created! Please Log In.")

    with tab3:
        st.markdown("Use temporary session (No Saving)")
        if st.button("Skip Login"):
            st.session_state.user_mode = "guest"
            st.session_state.user_username = "guest"
            cookie_manager.set("c_user", "guest", expires_at=pd.Timestamp.now() + pd.Timedelta(days=1))
            st.rerun()

if st.session_state.user_mode is None:
    # Always show the dialog if not logged in. No fallback screen.
    login_dialog()
    st.stop() # Ensure nothing else renders until logged in

# --- Settings Dialog ---
@st.dialog("Account Settings")
def settings_dialog():
    st.caption(f"Manage account for: **{st.session_state.user_username}**")

    st.subheader("🖼️ Change Avatar")
    new_av = st.text_input("Image URL", value=get_user_avatar(st.session_state.user_username))
    if st.button("Update Avatar"):
        update_user_avatar(st.session_state.user_username, new_av)
        st.success("Avatar Updated! Refreshing...")
        time.sleep(1)
        st.rerun()

    st.divider()

    st.subheader("🔑 Change Password")
    p1 = st.text_input("New Password", type="password", key="new_p1")
    p2 = st.text_input("Confirm Password", type="password", key="new_p2")
    if st.button("Update Password"):
        if len(p1) < 4: st.error("Too short")
        elif p1 != p2: st.error("Mismatch")
        else:
            save_user(st.session_state.user_username, p1, new_av) # Overwrites with new pass
            st.success("Password Changed!")

# --- Sidebar ---
# --- Sidebar Layout ---
with st.sidebar:
    # 1. Profile Section
    if st.session_state.user_mode == "user":
        u_av = get_user_avatar(st.session_state.user_username)

        # Centered Avatar (Compact 45px) with AGGRESSIVE negative margin
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; margin-top: -60px;">
                <img src="{u_av}" style="width: 45px; height: 45px; border-radius: 50%; object-fit: cover; margin-bottom: 2px; border: 2px solid #333;">
                <h4 style="margin: 0; padding: 0; font-size: 0.9rem;">{st.session_state.user_username}</h4>
            </div>
            """, unsafe_allow_html=True)

        # Profile Menu (Popover) - Minimal margin
        st.markdown("<div style='margin-bottom: 2px;'></div>", unsafe_allow_html=True)
        with st.popover("Manage Account", use_container_width=True): # Popover might still use this, usually button logs are specific. I'll stick to button first? No, error was generic. Safe to change.
            if st.button("⚙️ Settings", key="pop_sett", width="stretch"):
                settings_dialog()
            if st.button("🔒 Logout", key="pop_logout", width="stretch"):
                # Safe Logout: Delete specific keys instead of clearing all to avoid Dialog race conditions
                keys_to_drop = ["user_mode", "user_username", "use_gpu_toggle"]
                for k in keys_to_drop:
                    if k in st.session_state: del st.session_state[k]

                try: cookie_manager.delete("c_user")
                except: pass

                time.sleep(0.5) # Allow frontend to sync
                st.rerun()

        # Secure Filename
        safe_name = "".join(x for x in st.session_state.user_username if x.isalnum())
        DB_FILE = os.path.abspath(f"db_{safe_name}.xlsx")
    else:
        # Guest: Identical Structure for Alignment with AGGRESSIVE negative margin
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; margin-top: -60px;">
                <img src="{DEFAULT_AVATAR}" style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; margin-bottom: 4px; opacity: 0.8;">
                <h4 style="margin: 0; padding: 0; font-size: 1rem;">Guest User</h4>
            </div>
            """, unsafe_allow_html=True)

        with st.popover("Guest Options", use_container_width=True):
             if st.button("🔒 Return to Login", use_container_width=True):
                # Safe Logout for Guest
                keys_to_drop = ["user_mode", "user_username", "use_gpu_toggle"]
                for k in keys_to_drop:
                    if k in st.session_state: del st.session_state[k]

                try: cookie_manager.delete("c_user")
                except: pass

                time.sleep(0.5)
                st.rerun()

        DB_FILE = None

    st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #333;'>", unsafe_allow_html=True)

    # 2. Hardware
    # 2. Hardware / Performance
    gpu_available = False
    try:
        if torch.cuda.is_available(): gpu_available = True
    except: pass

    # Init Session State for GPU
    if "use_gpu_toggle" not in st.session_state:
        st.session_state.use_gpu_toggle = gpu_available

    def on_gpu_change():
        st.session_state.use_gpu_toggle = st.session_state.gpu_toggle_widget

    # Stacked Layout: Header -> Toggle -> Button
    st.markdown("**⚡ Performance**")

    # Toggle (Clean)
    st.toggle("GPU Acceleration", value=st.session_state.use_gpu_toggle, disabled=not gpu_available, key="gpu_toggle_widget", on_change=on_gpu_change, help="Enable if you have an NVIDIA GPU. Falls back to CPU otherwise.")

    # Status Text (Small)
    if not gpu_available:
        st.caption("⚠️ Cloud GPU unavailable")
    else:
        st.caption("✅ NVIDIA Enabled")

    # Clear RAM Button (Full Width)
    if st.button("🧹 Clear RAM", help="Clears Streamlit cache and GPU memory to prevent crashes.", key="clear_ram_button", use_container_width=True):
        st.cache_resource.clear()
        if gpu_available: torch.cuda.empty_cache()
        st.toast("RAM Cleared & Cache Reset")

    st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #333;'>", unsafe_allow_html=True)

    # 3. Configuration & Nav
    parsing_mode = st.radio("Extraction Engine", ["AI Mode (Smart)", "Fast Mode (Basic)"], help="AI Mode: Smart extraction (Requires Key). Fast Mode: improved regex (Free/Private).")
    
    if parsing_mode.startswith("AI Mode"):
        st.markdown("<hr style='margin: 5px 0; border-top: 1px solid rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
        hf_key = st.text_input("HuggingFace API Key", value="", type="password", help="Your private Hugging Face Access Token (Read Permission).")
        
        with st.expander("ℹ️ How to get a Free API Key?"):
            st.markdown("""
            1. **Sign Up/Login**: Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
            2. **Create Token**: Click 'New token', name it, select **'Read'** role.
            3. **Paste**: Copy the token starting with `hf_...` and paste it above.
            
            **💡 Pro Tip for Rate Limits:**
            If you see a `429 Rate Limit` error, the limit is per account. You can simply create a new **free** Hugging Face account to get a fresh limit immediately.
            """)
    else:
        hf_key = None

    # White Separator for Navigate
    st.markdown("<hr style='margin: 10px 0; border-top: 1px solid rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    page = st.radio("Navigate", ["📤 Extractor", "🗄️ Database History"], help="Switch views between the Extractor tool and your saved Database History.")

    # 4. Logout Removed (Now in Popover)

# --- Page 1 ---
if page == "📤 Extractor":

    # --- Header Layout ---
    st.markdown("<h1 style='text-align: center; margin: 0; padding: 0;'>AI BUSINESS CARD EXTRACTOR</h1>", unsafe_allow_html=True)

    # Row 2: Info Box
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 Upload business cards (Image/ZIP) to extract structured data.")

    # Restore State
    if "final_df" not in st.session_state:
        st.session_state.final_df = None

    if "file_uploader_key" not in st.session_state: st.session_state.file_uploader_key = 0

    # Full Width Uploader
    uploaded_files = st.file_uploader("Upload Cards", type=['png', 'jpg', 'jpeg', 'zip'], accept_multiple_files=True, key=f"uploader_{st.session_state.file_uploader_key}", help="Select images or a ZIP file containing business cards.")

    # Clear Button (Small, Right Aligned below uploader)
    # --- Manual Execution Flow ---
    if uploaded_files:
        # Determine State: Have we processed THESE files yet?
        # We use session_state.final_df as the indicator of "Done"

        # 1. Clear / Reset State (Button turns into Clear All after processing)
        # 1. Clear / Reset State (Button turns into Clear All after processing)
        if st.session_state.final_df is not None:
            if st.button("🗑️ Clear All", width="stretch"):
                st.session_state.file_uploader_key += 1
                st.session_state.final_df = None
                st.session_state.last_uploaded_names = set()
                st.rerun()
            # User request: "vanish" the button. 
            # We do NOT show "Clear All" here. We rely on the "Start New Extraction" button in the persisted block.
            pass

        # 2. Execute State
        else:
            if st.button("🚀 Execute Extraction", width="stretch", help="Start the AI extraction pipeline. This may take time.", type="primary"):

                if parsing_mode.startswith("AI Mode") and not hf_key:
                    st.error("Missing API Key"); st.stop()

                # Run Processing Logic Immediately
                with st.spinner("Initializing Models..."):
                    nlp_model = load_spacy_model()
                    reader = load_ocr_model(use_gpu=st.session_state.get("use_gpu_toggle", True))

                files = []
                current_file_names = set() 

                for f in uploaded_files:
                    files.append((f.name, f.getvalue()))
                    current_file_names.add(f.name)

                # Store names 
                st.session_state.last_uploaded_names = {f.name for f in uploaded_files}

                # OOM Safe Pipeline
                total = len(files)
                st.markdown(f"Scanning {total} files...")
                ocr_results = []
                bar = st.progress(0)

                # 1. Serial OCR
                for i, (fname, fbytes) in enumerate(files):
                    try:
                        img = preprocess_image_bytes(fbytes)
                        txt = reader.readtext(img, detail=0)
                        if torch.cuda.is_available(): torch.cuda.empty_cache() 
                        ocr_results.append({"fname": fname, "text": txt})
                    except Exception as e:
                        try:
                            print(f"GPU Failed for {fname}, switching to CPU... Error: {e}")
                            cpu_reader = load_ocr_model(use_gpu=False)
                            txt = cpu_reader.readtext(img, detail=0)
                            ocr_results.append({"fname": fname, "text": txt})
                        except Exception as e2:
                            ocr_results.append({"fname": fname, "error": str(e2)})
                    bar.progress((i+1) / (total*2))

                # 2. Parallel AI (Sequential)
                final_results = []
                def process_ai(item):
                    if "error" in item: return {"Image_File_Name": item["fname"], "Status": "ERROR", "Raw_Text": item["error"]}
                    try:
                        if parsing_mode.startswith("AI Mode"):
                            d = extract_with_llm(item["text"], hf_key, nlp_model=nlp_model)
                            # Check for API Errors propagated from extract_genai_pure
                            if "error" in d and ("INVALID_KEY" in d["error"] or "LIMIT_EXCEEDED" in d["error"]):
                                return d 
                        else:
                            d = parse_extracted_text_heuristic(item["text"], nlp_model)

                        d["Image_File_Name"] = item["fname"]
                        return d
                    except Exception as e:
                        return {"Image_File_Name": item["fname"], "Status": "Err", "Raw_Text": str(e)}

                st.markdown("Extracting Data (Sequential)...")
                api_error_stop = False
                for i, item in enumerate(ocr_results):
                    res = process_ai(item)

                    # Critical API Error Handling
                    if "error" in res and "INVALID_KEY" in res["error"]:
                        st.error(f"⚠️ {res['error']}"); api_error_stop = True; break
                    if "error" in res and "LIMIT_EXCEEDED" in res["error"]:
                        st.error(f"🛑 {res['error']}"); api_error_stop = True; break

                    final_results.append(res)
                    bar.progress(0.5 + ((i+1)/total * 0.5))
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                if not api_error_stop:
                    st.success("Done!")
                    df = pd.DataFrame(final_results)
                    
                    # Drop unwanted internal columns
                    cols_to_drop = ["Status", "Model_Used"]
                    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

                    st.markdown("### 📊 Extracted Data")
                    st.dataframe(df, use_container_width=True)
                    
                    # Excel Download (Full Data)
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)

                    # DB Save (Update DB File with FULL data, but user typically downloads from here)
                    if DB_FILE:
                        try:
                            old = pd.read_excel(DB_FILE) if os.path.exists(DB_FILE) else pd.DataFrame()
                            pd.concat([old, df], ignore_index=True).to_excel(DB_FILE, index=False)
                            st.success(f"✅ Data Saved to History: {os.path.basename(DB_FILE)}")
                        except PermissionError:
                             st.error(f"❌ Could not save to {os.path.basename(DB_FILE)}. Is the file open in Excel? Close it and re-run.")
                        except Exception as e: 
                             st.error(f"❌ Save Failed: {str(e)}")
                    else:
                        st.toast("Guest Mode: Data extracted but NOT saved to history.")

                    st.download_button("💾 Download Excel", buffer.getvalue(), "business_cards_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.session_state.final_df = df # Store full df for persistence

                    # Store current names to prevent re-execution loop
                    st.session_state.last_uploaded_names = current_file_names
                    st.rerun() # Force Button Swap immediately

    # Persisted Display - only show if NOT currently processing (to avoid double flash, though logic mostly handles this)
    # The user requested removing the "2nd table idea", but we need to show the RESULT if the page reloads.
    # We will just show final_df if it exists.
    # Persisted Display - Show if data exists
    if st.session_state.final_df is not None:
         st.markdown("### 📊 Extracted Data (Saved)")
         st.dataframe(st.session_state.final_df, use_container_width=True)

         # Action Buttons for Persisted State
         c1, c2 = st.columns(2)
         with c1:
             buffer = io.BytesIO()
             with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.final_df.to_excel(writer, index=False)
             st.download_button("💾 Download Excel", buffer.getvalue(), "business_cards_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_persisted", use_container_width=True)
         with c2:
             if st.button("🔄 Start New Extraction", key="clear_persisted", use_container_width=True):
                 st.session_state.file_uploader_key += 1
                 st.session_state.final_df = None
                 st.session_state.last_uploaded_names = set()
                 st.rerun()

# --- Page 2 ---
elif page == "🗄️ Database History":
    if st.session_state.user_mode == "guest":
        st.warning("🚫 History is not available in Guest Mode. Please Login to save cards.")
        st.info("You can reset the app from sidebar to login.")
    else:
        st.title(f"Database: {st.session_state.user_username}")
        if os.path.exists(DB_FILE):
            df = pd.read_excel(DB_FILE)
            # Drop unwanted internal columns for display
            cols_to_drop = ["Status", "Model_Used"]
            df_display = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
            st.dataframe(df_display, use_container_width=True)
            with open(DB_FILE, "rb") as f:
                st.download_button("Download Excel", f, "cards.xlsx")
        else:
            st.info("No saved data yet.")

# --- Footer (Main Page) ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>Made with ❤️ by <strong>Vedant Kalal</strong></p>
        <p>
            <a href="mailto:vedantkalal28@gmail.com" style="text-decoration: none;">📧 Email</a> | 
            <a href="https://www.linkedin.com/in/vedantkalal" style="text-decoration: none;">🔗 LinkedIn</a>
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)