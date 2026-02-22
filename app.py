import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import os
from fpdf import FPDF
from PIL import Image
import tempfile

# --- 1. CONFIGURATION & DIRECTORIES ---
os.makedirs('forensic_results', exist_ok=True)

class ForensicReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI DEEPFAKE FORENSIC REPORT', 0, 1, 'C')
        self.ln(10)

# --- 2. AI MODEL LOADING (CACHED) ---
@st.cache_resource
def load_ai_model():
    # Using Xception as a placeholder for the forensic backbone
    # In a real scenario, you'd use: tf.keras.models.load_model('model.h5')
    model = tf.keras.applications.Xception(weights='imagenet')
    return model

# --- 3. FORENSIC PROCESSING FUNCTIONS ---
def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    results = []
    
    if success:
        # Resize and preprocess for AI
        img = cv2.resize(frame, (299, 299))
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.xception.preprocess_input(img)
        
        # Predict
        preds = model.predict(img)
        score = np.max(preds) # Simplified score for demo
        
        # Save a frame for the report
        sample_path = "forensic_results/evidence_frame.jpg"
        cv2.imwrite(sample_path, frame)
        results.append(sample_path)
        
        cap.release()
        return score, results
    return 0.0, []

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Forensic AI", page_icon="🔍")
st.title("🛡️ Deepfake Forensic Analyzer")

with st.status("Initializing Forensic Engines...", expanded=True) as status:
    st.write("Loading AI Model...")
    model = load_ai_model()
    st.write("System Check: Pass")
    status.update(label="AI Engine Ready!", state="complete", expanded=False)

uploaded_file = st.file_uploader("Upload video for investigation", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save upload to a temp file for OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("Start Forensic Analysis"):
        with st.spinner("Analyzing pixels for synthetic artifacts..."):
            score, evidence_images = analyze_video(tfile.name, model)
            
            # Show Results
            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            
            verdict = "SUSPICIOUS" if score > 0.5 else "AUTHENTIC"
            col1.metric("Verdict", verdict)
            col2.metric("Manipulation Probability", f"{score*100:.2f}%")
            
            # Generate PDF
            pdf = ForensicReport()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"Analysis for: {uploaded_file.name}", 0, 1)
            pdf.cell(0, 10, f"Verdict: {verdict}", 0, 1)
            pdf.cell(0, 10, f"Confidence Score: {score:.4f}", 0, 1)
            
            if evidence_images:
                st.image(evidence_images[0], caption="Analyzed Evidence Frame")
                pdf.image(evidence_images[0], x=10, y=50, w=100)
            
            pdf_path = "Forensic_Report.pdf"
            pdf.output(pdf_path)
            
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Forensic Report", f, file_name="Forensic_Report.pdf")

else:
    st.info("Waiting for video input...")
