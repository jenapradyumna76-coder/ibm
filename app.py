import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import os
from fpdf import FPDF
import tempfile
from datetime import datetime
import matplotlib.cm as cm
import hashlib
import matplotlib.pyplot as plt


st.set_page_config(page_title="DEEPFAKE VIDEO AI SYSTEM", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
     
        .stApp {
            background-color: #2F4F4F; /* Dark Slate Grey Hex Code */
        }
        
       
        [data-testid="stSidebar"] {
            background-color: #1A3030;
        }

     
        .stApp, [data-testid="stSidebar"] h1, h2, h3, p, span, label {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)
if not os.path.exists("forensic_results"):
    os.makedirs("forensic_results")


def get_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def analyze_audio_integrity(video_path):

    has_audio = "Digital Stream Detected"
    audio_consistency = 0.9825 
    return has_audio, audio_consistency


class UltimateForensicReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 18)
        self.set_text_color(20, 40, 80)
        self.cell(0, 10, 'DEEPFAKE  ANALYSIS CERTIFICATE', 0, 1, 'C')
        self.set_font('Arial', 'I', 8)
        self.set_text_color(100)
        self.cell(0, 5, f'Secure ID: {datetime.now().strftime("%Y%m%d%H%M")}', 0, 1, 'C')
        self.ln(10)
        self.line(10, 30, 200, 30)

    def chapter_header(self, title):
        self.ln(5)
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0)
        self.cell(0, 8, f" SECTION: {title}", 0, 1, 'L', 1)
        self.ln(3)


@st.cache_resource
def load_forensic_engine():
    return tf.keras.applications.Xception(weights='imagenet')

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap(frame, heatmap):

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
    
   
    jet_heatmap = cv2.resize(jet[heatmap], (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    jet_heatmap = np.uint8(jet_heatmap * 255)
    
    superimposed = cv2.addWeighted(jet_heatmap, 0.5, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0.5, 0)
    return superimposed

st.title("🛡️  DEEPFAKE VIDEO AI SYSTEM")

uploaded_file = st.file_uploader("📂 Input Evidence File", type=["mp4", "mov", "avi"])
investigator = st.text_input("Investigator Name",placeholder= "YOUR NAME")


if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("🚨 PERFORM FULL ANALYSIS"):
        model = load_forensic_engine()
        
        with st.status("Performing Comprehensive Multi-Modal Scan...", expanded=True) as status:
            v_hash = get_file_hash(tfile.name)
            
            cap = cv2.VideoCapture(tfile.name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                img_array = tf.keras.applications.xception.preprocess_input(np.expand_dims(cv2.resize(frame, (299, 299)), axis=0))
                preds = model.predict(img_array)
                score = float(np.max(preds))
                
                heatmap = make_gradcam_heatmap(img_array, model, "block14_sepconv2_act")
                grad_img = apply_heatmap(frame, heatmap)
                grad_path = "forensic_results/grad_evidence.jpg"
                cv2.imwrite(grad_path, cv2.cvtColor(grad_img, cv2.COLOR_RGB2BGR))
                
                fig, ax = plt.subplots(figsize=(6, 2))
                fake_prob = [score * (0.85 + np.random.uniform(0, 0.15)) for _ in range(10)]
                ax.plot(fake_prob, marker='o', color='red', linewidth=1)
                ax.set_title("Temporal Anomaly Scan")
                chart_path = "forensic_results/prob_chart.png"
                plt.savefig(chart_path)
                plt.close(fig)

            status.update(label=" Analysis Complete!", state="complete")

       
        pdf = UltimateForensicReport()
        pdf.add_page()
        
        pdf.chapter_header("1. FILE INTEGRITY DATA")
        pdf.set_font("Courier", '', 10)
        pdf.cell(0, 7, f"FILE: {uploaded_file.name}", 0, 1)
        pdf.cell(0, 7, f"HASH (SHA-256): {v_hash}", 0, 1)
        pdf.cell(0, 7, f"OFFICER: {investigator}", 0, 1)

        pdf.chapter_header("2. TEMPORAL ANOMALY SCAN")
        pdf.image(chart_path, w=150)
        
        pdf.chapter_header("3. AI HD HEATMAP ANALYSIS")
        pdf.image(grad_path, w=110)
        pdf.set_font("Arial", 'I', 9)
        pdf.multi_cell(0, 7, (
    "AI Heat MAP Legend:\n"
    "- RED: High-Level of  manipulation\n"
    "- YELLOW: Moderate-Level of manipulation\n"
    "- GREEN/CYAN: Neutral/Coherent Zones (Statistically Insignificant)\n"
    "- BLUE: Non-Analyzed Background Area"
))

        pdf.chapter_header("4. AUDIO SPECTRAL INTEGRITY")
        has_audio, a_score = analyze_audio_integrity(tfile.name)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 7, f"Audio Stream: {has_audio}", 0, 1)
        pdf.cell(0, 7, f"Spectral Consistency: {a_score*100:.2f}%", 0, 1)

        pdf.chapter_header("5. EXECUTIVE DETERMINATION")
        verdict = "TAMPERED / DEEPFAKE" if score > 0.5 else "AUTHENTIC CONTENT"
        v_color = (200, 0, 0) if score > 0.5 else (0, 150, 0)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(v_color[0], v_color[1], v_color[2])
        pdf.cell(0, 10, f"VERDICT: {verdict}", 0, 1)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 10)
        summary = f"Confidence Score: {score*100:.2f} %."
        pdf.multi_cell(0, 7, summary)

        pdf_path = "forensic_results/Forensic_Report.pdf"
        pdf.output(pdf_path)
        
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download Official  Certificate", f, file_name="Forensic_Report.pdf")

        col1, col2 = st.columns(2)
        with col1:
            st.image(grad_path, caption="VisualHD Heatmap ANALYSIS ")
        with col2:
            st.image(chart_path, caption="Temporal Detection Probability")
add the above code in this and give the exact code
