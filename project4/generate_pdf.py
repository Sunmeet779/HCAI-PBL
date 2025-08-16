from fpdf import FPDF
import datetime

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
            self.multi_cell(0, 8, f"- {bullet}")
        self.ln(5)

def generate_method_study_pdf(path):
    pdf = PDF()
    pdf.add_page()
    
    # Task 1: Enhanced method description
    pdf.chapter("Task 1: Guided Active Learning Recommender", [
        "Matrix Factorization with cold-start support (K=20 latent factors)",
        "Real-time prediction visualization showing how each rating affects recommendations",
        "Genre-aware recommendations that consider movie categories",
        "Interactive UI shows similar movies and predicted impact of each rating",
        "Users see immediate feedback on how their ratings influence recommendations",
        "Algorithm combines collaborative filtering with content-based features"
    ])
    
    # Task 2: Detailed user study design
    pdf.chapter("Task 2: User Study Design", [
        "Hypothesis: Showing prediction impacts improves user engagement and recommendation quality",
        "Study Design: A/B testing with two groups (with/without prediction visualization)",
        "Metrics: Completion rate, time spent, rating diversity, recommendation accuracy",
        "Participants: 100 users recruited via university mailing lists and social media",
        "Procedure:",
        "  1. Pre-study questionnaire about movie preferences",
        "  2. Random assignment to control or test group",
        "  3. Rating interface with/without prediction visualization",
        "  4. Post-study questionnaire about user experience",
        "  5. Optional follow-up interview with selected participants",
        "Analysis: Compare metrics between groups using t-tests and ANOVA",
        "Ethics: Informed consent, data anonymization, right to withdraw"
    ])
    
    # Task 3: Interface description
    pdf.chapter("Task 3: Study Interface Implementation", [
        "Landing page with study information and participation options",
        "Interactive rating interface with real-time feedback",
        "Visualization of how ratings affect recommendations",
        "Genre information and similar movies display",
        "Responsive design works on desktop and mobile devices",
        "Progress indicators and clear instructions",
        "Option to download results or continue later"
    ])
    
    pdf.output(path)