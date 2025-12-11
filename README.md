🚀 **AI Business Card Extractor** 📊
================================

> "Instantly convert Business Cards into actionable Excel data using AI & OCR" 💡

📖 Description
--------------

The **AI Business Card Extractor** is a powerful Python-based tool designed to digitize business cards with high accuracy. By combining **EasyOCR** for text detection and **Llama 3.1** (via Hugging Face) for intelligent parsing, this application solves the problem of manual data entry. It organizes messy, unstructured text from card images into structured data (Name, Phone, Email, Company, Address) ready for Excel export.

The system features a **Streamlit** interface that is both intuitive and visually appealing. It supports **bulk processing** (uploading multiple images or ZIP files), making it ideal for sales professionals and businesses. Security is handled via a local JSON-based user database with hashed passwords, offering both safe Login and Guest access modes.

Whether you have a stack of physical cards or a folder of digital scans, the AI Business Card Extractor streamlines your contact management workflow.

✨ Features
-----------

The following are some of the key features of the system:

1. **AI-Powered Extraction**: Uses `meta-llama/Llama-3.1-8B-Instruct` to intelligently parse and un-jumble OCR text 🤖
2. **Bulk Processing**: Upload individual images or entire ZIP files for batch extraction 📂
3. **Dual Parsing Modes**:
   - **AI Mode**: Smartest extraction using GenAI (Requires free Hugging Face Key).
   - **Fast Mode**: Privacy-focused, heuristic-based extraction (No API key needed) ⚡
4. **User Management**: Secure Signup/Login with password hashing and Avatar customization 🔐
5. **Persistent History**: Automatically saves extracted data to a personal user database (`db_{username}.xlsx`) 🗄️
6. **Smart Preprocessing**: Auto-enhances images (denoising, sharpening) for 1080p OCR optimization 🖼️
7. **Excel Export**: One-click download of all extracted data in `.xlsx` format 📊
8. **GPU Acceleration**: Built-in toggle to utilize NVIDIA CUDA for faster OCR inference 🚀

🧰 Tech Stack Table
-------------------

| Component               | Technology                                   |
| ----------------------- | -------------------------------------------- |
| **Frontend**      | Streamlit (Web UI)                           |
| **Backend**       | Python 3.x                                   |
| **OCR Engine**    | EasyOCR, OpenCV (Image Processing)           |
| **AI/LLM**        | LangChain, Hugging Face Endpoint (Llama 3.1) |
| **NLP**           | spaCy (Named Entity Recognition)             |
| **Data Handling** | Pandas, OpenPyXL                             |
| **Database**      | JSON-Based (Users), Excel (Data Storage)     |

📁 Project Structure
--------------------

The project is organized into the following folders and files:

* `app.py`: The main application orchestrator containing UI, Authentication, and Extraction logic 📊
* `users_db.json`: Secure JSON storage for user credentials and profiles 🔐
* `requirements.txt`: List of Python dependencies required to run the app 📦
* `README.md`: Project documentation 📄

⚙️ How to Run
---------------

To run the AI Business Card Extractor, follow these steps:

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/vedant-kalal/AI-VISITING-CARD-EXTRACTOR.git
   ```
2. Navigate to the project directory:
   ```bash
   cd AI-VISITING-CARD-EXTRACTOR
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   *(Note: Ensure you have PyTorch installed compatible with your system)*

### Environment

1. Create a variable `hf_token` if you want to hardcode it, usually managed via UI input for security.
2. Optional: Create a `.env` file if you wish to store environment secrets.

### Run

1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. The app will open in your default browser at `http://localhost:8501`.

🧪 Testing Instructions
-----------------------

1. **Login**: Create a new account or use "Guest Mode".
2. **API Key**: If using AI Mode, enter your free Hugging Face Read Token (Guide available in app).
3. **Upload**: Drag and drop a sample business card image.
4. **Execute**: Click "Execute Extraction" and watch the logs.
5. **Verify**: Check the results table and download the Excel file.

📸 Screenshots
--------------

*(Add text description or placeholders for screenshots here)*

* **Login Screen**: Sleek dark-mode authentication.
* **Extraction Dashboard**: File uploader and real-time progress bars.
* **Results Table**: Clean, structured data view with download options.

📦 API Reference
----------------

* **Hugging Face Inference API**: Used for Llama 3.1 8B interactions.
* **EasyOCR**: Used for optical character recognition.

👤 Author
---------

The AI Business Card Extractor was developed by [Vedant Kalal](https://www.linkedin.com/in/vedantkalal) 🙋‍♂️

📝 License
----------

The AI Business Card Extractor is licensed under the [MIT License](https://opensource.org/licenses/MIT) 📜
Copyright (c) 2024 [Vedant Kalal](https://github.com/vedantkalal)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 📜
