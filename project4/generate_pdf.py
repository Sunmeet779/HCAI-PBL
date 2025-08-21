import datetime
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(0, 51, 102)  # Dark blue
        self.cell(0, 10, "Project 4: Method & User Study Summary", align='L')
        self.cell(0, 10, f"Page {self.page_no()}", align='R')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)  # Gray
        self.cell(0, 10, f"Page {self.page_no()}", align='C')

    def add_title_page(self):
        self.add_page()
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(0, 51, 102)  # Dark blue
        self.cell(0, 20, "Project 4: Method & User Study Summary", ln=True, align='C')
        self.set_font("Helvetica", "I", 16)
        self.cell(0, 15, "Recommender Systems and Active Learning", ln=True, align='C')
        self.ln(20)
        self.set_font("Helvetica", "", 12)
        self.ln(20)
        # Table of Contents on the same page
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, "Table of Contents", ln=True, align='C')
        self.ln(10)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(0, 0, 0)
        sections = [
            ("Project Overview: Influence of Future Predictions", 1),
            ("Task 1: Guided Active Learning", 1),
            ("Task 2: User Study Design and Evaluation", 1),
            ("Task 3: Study Interface Implementation", 1)
        ]
        for title, page in sections:
            self.cell(0, 8, title, ln=False)
            self.cell(0, 8, str(page), ln=True, align='R')
        self.ln(10)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(0, 51, 102)
        self.ln(8)
        self.multi_cell(0, 10, title)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def chapter(self, title, bullets):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 102, 204)  # Lighter blue
        self.multi_cell(0, 10, title)  # Text wrapping for long titles
        self.ln(5)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(0, 0, 0)
        for bullet in bullets:
            self.cell(5)  # Indent
            self.multi_cell(0, 8, f"- {bullet}")
        self.ln(5)

def generate_method_study_pdf(path):
    pdf = PDF()
    pdf.set_margins(left=20, top=20, right=20)
    
    # Title page
    pdf.add_title_page()
    
    # Project Introduction and Motivation
    pdf.add_page()  # Start content on a new page after title
    pdf.chapter("Project Overview: Influence of Future Predictions on Active Learning in Recommender Systems", [
        "This project addresses the cold-start problem in recommender systems, where new users have no prior ratings and the system must quickly learn their preferences.",
        "We use the MovieLens dataset (100k ratings, 9k movies, 600 users) to build a movie recommender that adapts to new users by interactively eliciting their tastes.",
        "Users rate movies on a scale from 0.5 to 5, indicating either their actual rating or their interest level if they haven't seen the movie.",
        "Unlike standard active learning, our method provides users with real-time feedback on how each rating will affect future recommendations, encouraging strategic and informed responses."
    ])

    # Task 1: Guided Active Learning Method
    pdf.chapter("Task 1: Guided Active Learning for Cold-Start Recommendation", [
        "Matrix factorization is performed using Singular Value Decomposition (SVD) on the user-item rating matrix, with K=50 latent factors (see model_training.py).",
        "Collaborative filtering is used: user (U) and movie (V) matrices are learned to minimize the error between predicted and actual ratings, with regularization applied during SVD.",
        "For new users, their latent representation is initialized and updated as they provide ratings, using the pre-trained movie factors (V) and regularized least squares (see recommender.py).",
        "Genre information from movies.csv is available for recommendations, but tags.csv is not actively used in the current implementation.",
        "Movie metadata (movie_metadata.json) is used to display genre and similar movies in the interface.",
        "Backend logic in recommender.py and views.py handles rating submissions, updates user factors, and generates recommendations in real time.",
        "The interface (study_interface.html, script.js) allows users to rate movies and see updated recommendations, but does not visualize the impact of each rating in detail.",
        "Data processing, model training, and prediction logic are implemented in Python and Django, with numpy used for matrix operations and storage (user_factors.npy, movie_factors.npy)."
    ])

    # Task 2: User Study Design
    pdf.chapter("Task 2: User Study Design and Evaluation", [
        "The study interface allows users to rate movies and records their ratings for later analysis.",
        "Metrics such as rating diversity and completion rate can be inferred from the ratings data exported from the system."
    ])

    # Task 3: Study Interface
    pdf.chapter("Task 3: Study Interface Implementation", [
        "The landing page (index.html) introduces the study, provides access to documentation, and allows users to start the study.",
        "The rating interface (study_interface.html) presents movies with genre information and allows users to submit ratings.",
        "The interface is responsive and provides clear instructions for users.",
        "All interface logic is implemented using Django templates, static CSS/JS, and backend views for integration and user experience."
    ])

    pdf.output(path)

if __name__ == "__main__":
    generate_method_study_pdf("static/project4/method_and_study.pdf")