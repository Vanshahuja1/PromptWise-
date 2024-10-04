import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
import fitz  # PyMuPDF
import os
import re
from langdetect import detect
import easyocr
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(page_title="🌍 PolySummarize: Multilingual Text Summarizer", page_icon="📝", layout="wide")

# Load Models
@st.cache_resource
def load_model():
    model_directory = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_directory)
    tokenizer = T5Tokenizer.from_pretrained(model_directory)
    return model, tokenizer

model, tokenizer = load_model()

@st.cache_resource
def load_translation_models():
    translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    return translation_model, translation_tokenizer

translation_model, translation_tokenizer = load_translation_models()

# Functions for translation, preprocessing, and summarization
def translate_text(text, src_lang):
    src_lang = src_lang.lower()
    if src_lang == "zh-cn":
        src_lang = "zh"
    translation_input = translation_tokenizer.prepare_seq2seq_batch([text], src_lang=src_lang, tgt_lang="en", return_tensors="pt")
    translated_ids = translation_model.generate(**translation_input)
    translated_text = translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

def preprocess_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

def summarize_text(text, prompts, summary_length):
    cleaned_text = preprocess_text(text)
    combined_text = f"summarize: {cleaned_text}"
    if prompts:
        combined_text += " " + " ".join(prompts)

    tokenized_text = tokenizer.encode(combined_text, return_tensors="pt", max_length=512, truncation=True, padding=True)

    if summary_length == "Short":
        max_length = 100  # Adjust as needed
    elif summary_length == "Medium":
        max_length = 200  # Adjust as needed
    else:  # Long
        max_length = 300  # Adjust as needed

    summary_ids = model.generate(tokenized_text, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Functions for file reading and OCR
def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def read_image(file, lang):
    image = Image.open(file)
    image_np = np.array(image)
    if lang in ['ja', 'ko', 'zh-cn', 'zh-tw']:
        reader = easyocr.Reader(['ja', 'ko', 'zh-cn', 'zh-tw', 'en'])
    else:
        reader = easyocr.Reader(['en', 'fr', 'de', 'es', 'it', 'pt'])
    return ' '.join(reader.readtext(image_np, detail=0))

def detect_language(text):
    return detect(text)

# UI Layout based on the provided image

st.title("🌍 PolySummarize: Multilingual Text Summarizer")
st.write("**Summarize your text across multiple languages with ease**. You can input text, upload PDFs, or image files, and get a summarized version in no time!")

# Define layout using columns to match the provided structure
header = st.columns([3, 1])
with header[0]:
    st.subheader("📑 **Choose Input Method**")
with header[1]:
    input_method = st.selectbox("Select input method", ["Direct Text Input", "Upload File"])

# Main layout
main_layout = st.columns([3, 1])  # Text input on the left, prompt and button on the right

# Text input area
with main_layout[0]:
    if input_method == "Direct Text Input":
        user_input = st.text_area("✍️ Enter your text here:", placeholder="Type or paste your text here...", height=200)
        file_text = user_input if user_input else None
    else:
        uploaded_file = st.file_uploader("📂 Upload your file (PDF, TXT, Image)", type=["pdf", "txt", "png", "jpg", "jpeg"])
        if uploaded_file:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".pdf":
                file_text = read_pdf(uploaded_file)
            elif ext == ".txt":
                file_text = read_txt(uploaded_file)
            else:
                temp_image_text = read_image(uploaded_file, 'en')
                detected_lang = detect_language(temp_image_text)
                file_text = read_image(uploaded_file, detected_lang)
        else:
            file_text = None

# Prompt and Summary button on the right
with main_layout[1]:
    st.subheader("💡 **Add Prompt**")
    if "prompts" not in st.session_state:
        st.session_state.prompts = []

    prompt = st.text_input("Add a focus prompt (e.g., 'highlight key points')")

    if st.button("➕ Add Prompt"):
        st.session_state.prompts.append(prompt)

    summary_length = st.selectbox("📏 Select Summary Length:", ["Short 📝", "Medium ✏️", "Long 📜"])

    if st.button("🔄 Generate Summary"):
        if file_text:
            summary = summarize_text(file_text, st.session_state.prompts, summary_length)
            st.subheader("📄 Summary")
            st.write(summary)
        else:
            st.error("Please input or upload a file to summarize!")

# Summary section below input and button
if file_text:
    detected_language = detect_language(file_text)
    st.write(f"**🌍 Detected Language:** {detected_language.capitalize()}")

    if detected_language != "en":
        if st.checkbox("🌐 Translate to English"):
            file_text = translate_text(file_text, detected_language)
            st.write("**📝 Translated Text**")
            st.text_area("Translated Text", value=file_text, height=150)

# Add professional styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e0eafc, #cfdef3);
        font-family: 'Arial', sans-serif;
    }
    .stTextArea, .stTextInput, .stButton, .stSelectbox {
        border-radius: 2px;
        border: 1px solid #ccc;
        box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stTextArea:hover, .stTextInput:hover, .stButton:hover, .stSelectbox:hover {
        border-color: #007BFF;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        padding: 10px 20px;
        border-radius: 2px;
        font-size: 16px;
    }
    .stSelectbox>div {
        font-size: 14px;
        color: #444;
    }
    </style>
""", unsafe_allow_html=True)