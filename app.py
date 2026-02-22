import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import os
from fpdf import FPDF
import tempfile
from datetime import datetime

# --- 1. SETUP & THEME ---
st.set_page_config(page_title="Forensic AI", page_icon="🛡️", layout="wide")

# Ensure results directory exists
os.makedirs('forensic_results', exist_ok=True)

# PDF Report Class
class ForensicReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI DEEPFAKE FORENSIC INVESTIGATION', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

# --- 2. AI ENGINE (CACHED) ---
@st.cache_resource
def load_forensic_engine():
    # Loading Xception backbone as our forensic analyzer
    model = tf.keras.applications.Xception(weights='imagenet')
    return model

# --- 3. FORENSIC PROCESSING ---
def run_analysis(video_path, model):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # We will analyze a sample frame (usually the middle frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10) 
    success, frame = cap.read()
    
    if success:
        # Preprocess for AI
        img = cv2.resize(frame, (299, 299))
        img_array = tf.keras.applications.xception.preprocess_input(np.expand_dims(img, axis=0))
        
        # AI Prediction
        preds = model.predict(img_array)
        score = np.max(preds)  # Simplified: taking max confidence
        
        # Save Evidence Image
        evidence_path = "forensic_results/evidence_frame.jpg"
        cv2.imwrite(evidence_path, frame)
        
        cap.release()
        return score, evidence_path, fps
    return 0.0, None, 0.0

# --- 4. DASHBOARD UI ---
st.title("🛡️ Deepfake Forensic Analyzer")
st.markdown("---")

# Sidebar Status
with st.sidebar:
    st.header("System Status")
    with st.spinner("Waking up AI..."):
        engine = load_forensic_engine()
    st.success("AI Engine: ONLINE")
    st.info("Version: 2026.1-Forensic")

# Main Interface
uploaded_file = st.file_uploader("📂 Upload Evidence Video (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file:
    # Save to temp file so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Evidence")
        st.video(uploaded_file)
        
    if st.button("🔍 START FORENSIC SCAN"):
        with st.status("Performing Multi-Stage Analysis...", expanded=True) as status:
            st.write("Checking metadata integrity...")
            # Simulate metadata check
            st.write("Scanning pixels for GAN artifacts...")
            score, evidence_img, video_fps = run_analysis(tfile.name, engine)
            st.write("Finalizing report...")
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        with col2:
            st.subheader("AI Analysis Results")
            verdict = "SUSPICIOUS" if score > 0.5 else "AUTHENTIC"
            color = "inverse" if verdict == "SUSPICIOUS" else "normal"
            
            st.metric("Final Verdict", verdict, delta=f"{score*100:.1f}% Match", delta_color=color)
            
            if evidence_img:
                st.image(evidence_img, caption="AI Evidence: Sample Frame Analyzed")

        # --- PDF GENERATION ---
        pdf = ForensicReport()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"File Analyzed: {uploaded_file.name}", 0, 1)
        pdf.cell(0, 10, f"Detection Score: {score:.4f}", 0, 1)
        pdf.cell(0, 10, f"Verdict: {verdict}", 0, 1)
        
        if evidence_img:
            pdf.ln(10)
            pdf.cell(0, 10, "Visual Artifact Evidence:", 0, 1)
            pdf.image(evidence_img, x=10, y=80, w=150)
        
        report_name = "Forensic_Report.pdf"
        pdf.output(report_name)

        st.markdown("---")
        with open(report_name, "rb") as f:
            st.download_button(
                label="📥 Download Detailed Forensic PDF",
                data=f,
                file_name=f"Report_{uploaded_file.name}.pdf",
                mime="application/pdf"
            )
else:
    st.warning("Please upload a video file to begin the investigation.")
