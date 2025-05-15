# ChilledExamPrep 🎓📚

**Created by: djmahe4**

Welcome to ChilledExamPrep! This Streamlit application is designed to be your ultimate study assistant, leveraging the power of Google's Gemini API and YOLO object detection to help you prepare for exams more effectively.

**Online Streamlit App:**

You can try out the live version of this application (running on Python 3.10) here:  
[**ChilledExamPrep**](https://chilledexamprep2.streamlit.app/) 🚀  

## ✨ Features

- **🏠 Unified Home Page:**
  - **API Key Configuration:** Securely input your Gemini API key to power the AI features.
  - **Productivity Tools:**
    - 🍅 **Pomodoro Timer:** Manage your study sessions with work/break intervals.
    - 💡 **Text-Based Mindmap:** Organize your thoughts and concepts hierarchically.

- **📝 Notes & Diagram Analysis:**
  - Upload your study notes in TXT or PDF format.
  - View and skim through your notes content.
  - **AI-Powered Q&A:** Ask questions about your notes and get answers from Gemini.
  - **Diagram Detection & Explanation (YOLOv8):**
    - Automatically extracts images from PDF notes.
    - Utilizes a [YOLO model](https://github.com/abinthm/Diagram-extractor-model/blob/main/runs/detect/train/weights/best.pt) (`diagram_det.pt` or default `yolov8n.pt`) to identify potential diagrams.
    - Performs OCR on detected diagrams to extract text.
    - Uses Gemini to explain concepts from diagrams.
  - **Downloadable Data:** Export processed notes text and diagram information in JSON format for training or analysis.

- **📜 Question Papers Chat:**
  - Upload previous year question papers (TXT or PDF).
  - Chat with Gemini using question paper context for better doubt resolution.

- **🧠 Concept Learning Aids:**
  - **Mnemonic Generation:** Gemini helps you memorize complex topics.
  - **Topic Simplification:** Get simplified explanations or analogies for hard topics.

## 🛠️ Tech Stack & Libraries

- **Streamlit:** Interactive web app framework.
- **Python 3.10:** (for online deployment)
- **Gemini API (Google Generative AI):** Model used - `gemini-1.5-flash`
- **Ultralytics YOLOv8:** For object detection
  - GitHub: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **PyMuPDF (fitz):** PDF parsing and image extraction
- **Pytesseract & Pillow (PIL):** OCR for diagram text
- **Standard Python Libraries:** `json`, `time`, `os`, `io`, `functools`, `numpy`

## ⚙️ Local Setup & Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/djmahe4/ChilledExamPrep.git
   cd ChilledExamPrep
   ```
2. **Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Tesseract OCR Engine**

Linux:
```bash
sudo apt-get install tesseract-ocr
```
macOS:
```bash
brew install tesseract
```
Windows:

Download from UB Mannheim Tesseract
Add path (e.g., C:\Program Files\Tesseract-OCR) to System PATH



5. **(Optional) Custom Diagram Detection Model**

Place your custom diagram_det.pt in the project root. Otherwise, yolov8n.pt will be used.

6. **Install dependencies (preferably after activation of virtual environment)**
   ```bash
   pip install -r requirements.txt
   ```

6. **Get Your Gemini API Key**

Get it from [Google AI Studio](https://aistudio.google.com/app/apikey)


7. **Run the Streamlit App**

streamlit run Study_App_Home.py

Open the app in your browser and enter your Gemini API key.


## 💡 Future Enhancements / To-Do

- [ ] Pomodoro timer with browser notifications & sound

- [ ] Graphical mindmap using streamlit-agraph or JS

- [ ] Error handling & logging improvements

- [ ] Save/load full study sessions (files, chats)

- [ ] Tag/categorize notes and diagrams

- [ ] Document/chat summarization

- [ ] User accounts & persistent storage

- [ ] JSON-to-NN pipeline examples (Colab, etc.)

- [ ] Diagram explanation feedback loop

- [ ] Gemini prompting fine-tuning for diagrams

- [ ] Performance optimization for large PDFs

- [ ] Theme customization options


## Acknowledgements

1. Streamlit

2. Google Gemini API

3. Ultralytics - YOLOv8

4. Developers of PyMuPDF, Pytesseract, Pillow, and all supporting libraries

5. [Abin Thomas](https://github.com/abinthm/)
