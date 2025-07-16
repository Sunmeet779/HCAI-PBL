# project4/generate_pdf.py
from fpdf import FPDF
import os, datetime

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Project 4: Method & User Study Summary", ln=True, align='C')
        self.set_font("Helvetica", "", 10)
        self.cell(0, 8, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
        self.ln(10)

    def chapter(self, title, bullets):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 10, title, ln=True)
        self.set_font("Helvetica", "", 11)
        for bullet in bullets:
            self.cell(10)
            self.cell(0, 8, f"- {bullet}", ln=True)  # REPLACED '•' with '-'
        self.ln(5)

def generate_method_study_pdf(path):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter("Task 1: Guided Recommender", [
        "Matrix Factorization with cold-start support",
        "Latent dimension K = 20",
        "Regularization lambda = 0.1",
        "Shows user influence on future predictions",
    ])
    pdf.chapter("Task 2: User Study Design", [
        "A/B between-subjects: see-impact vs. no-impact",
        "Metrics: completion, satisfaction, diversity",
        "Recruitment: mailing lists, social media"
    ])
    pdf.chapter("Task 3: Study Interface", [
        "User rates 10 movies from the database",
        "UI updates top-N recommendations live",
        "Optional: show how scores would change"
    ])
    pdf.output(path)
