import os
from fpdf import FPDF
# Assuming your previous scripts are saved as metadata_utils.py and forensic_ai.py
# from metadata_utils import analyze_metadata
# from forensic_ai import run_forensic_analysis

class ForensicReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI DEEPFAKE FORENSIC ANALYSIS REPORT', 0, 1, 'C')
        self.ln(10)

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, content)
        self.ln(5)

def generate_final_report(video_name, ai_score, metadata_flags, images_folder):
    pdf = ForensicReport()
    pdf.add_page()

    # 1. Summary Section
    status = "SUSPICIOUS" if ai_score > 0.5 or metadata_flags else "AUTHENTIC"
    pdf.add_section("1. EXECUTIVE SUMMARY", 
                    f"Video Analyzed: {video_name}\n"
                    f"Overall Verdict: {status}\n"
                    f"AI Confidence Score: {ai_score*100:.2f}% (Probability of Manipulation)")

    # 2. Metadata Section
    meta_text = "\n".join(metadata_flags) if metadata_flags else "No suspicious metadata found."
    pdf.add_section("2. DIGITAL METADATA ANALYSIS", meta_text)

    # 3. Visual Evidence Section (Images from Grad-CAM)
    pdf.add_section("3. VISUAL ARTIFACTS (Grad-CAM)", 
                    "The following frames highlight areas where the AI detected synthetic patterns:")
    
    # Grab the first 3 evidence images from your Grad-CAM folder
    evidence_images = [f for f in os.listdir(images_folder) if f.endswith('.jpg')][:3]
    for img_name in evidence_images:
        img_path = os.path.join(images_folder, img_name)
        pdf.image(img_path, w=100)
        pdf.ln(5)

    pdf.output("Forensic_Investigation_Report.pdf")
    print("Report generated: Forensic_Investigation_Report.pdf")

# --- EXECUTION FLOW ---
# 1. metadata = analyze_metadata("test_video.mp4")
# 2. run_forensic_analysis("test_video.mp4") 
# 3. generate_final_report("test_video.mp4", 0.89, ["Lavf encoder detected", "Missing Camera Info"], "forensic_results"