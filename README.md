# Human-Centeric AI — Project-Based Learning (HCAI-PBL)

---

## 🚀 Deployed Project

Check out the deployed project here: [https://hcai-pbl-production.up.railway.app/](https://hcai-pbl-production.up.railway.app/)

---

## 📋 Project Overview

This repository hosts a suite of Django apps that demonstrate various human-centered AI techniques, including dynamic interfaces, interactive ML workflows, active learning for text, model explainability, personalized movie recommendations, and reinforcement learning with human feedback.

---

## 🔬 Methodology

The project is structured as a collection of five Django applications, each focusing on a specific AI technique. The methodology includes:

- **Data Utilization:** Leveraging datasets like IMDB 50k, MovieLens, and Palmer Penguins for real-world relevance.
- **Model Development:** Implementing supervised learning, active learning, explainable AI, collaborative filtering, and reinforcement learning with human feedback (RLHF).
- **Web Integration:** Using Django to create responsive, user-friendly interfaces for model interaction and visualization.
- **Optimization:** Addressing deployment constraints (e.g., Heroku's 500MB slug size limit) by excluding large files and integrating external storage solutions.

---

## 📂 Project Structure

### 🏠 **Home** — Central Navigation

**Purpose:** Provides a landing page for navigating the project's applications.

**Key Files:**
- `templates/home/index.html`: Renders the main navigation interface.
- `home/static/home/`: Contains CSS, JavaScript, and images for styling.

**Outcome:** A professional, responsive hub that ensures seamless access to all project components.

### 📊 **Project 1** — Data Visualization and Machine Learning

**Purpose:** Facilitates the upload, visualization, and training of supervised machine learning models using user-provided CSV datasets.

**Key Files:**
- `views.py`: Manages HTTP requests and model training logic.
- `forms.py`: Handles CSV file uploads.
- `utils.py`: Implements data preprocessing and model utilities.
- `templates/upload_train.html`: Provides an interface for data exploration and predictions.

**Outcome:** Enables end-to-end machine learning workflows with interactive data visualization and real-time predictions.

### 📝 **Project 2** — Active Learning for Text Classification

**Purpose:** Applies active learning to train sentiment classifiers on the IMDB 50k movie review dataset, optimizing sample selection for efficiency.

**Key Files:**
- `data/IMDB Dataset.csv`: Dataset for sentiment analysis.
- `templates/project2/project2_home.html`: Displays analysis results with custom filters.
- `utils.py`: Implements active learning algorithms and template filters.

**Outcome:** Demonstrates high-accuracy sentiment classification with minimal labeled data, enhancing efficiency in data annotation.

### 🔍 **Project 3** — Interactive Explainable AI

**Purpose:** Explores model interpretability using the Palmer Penguins dataset, with visualizations for decision trees, logistic regression, and counterfactual analysis.

**Key Files:**
- `logic/logic_explainer.py`: Implements explainability algorithms.
- `templates/project3/`: Renders interactive visualizations for model explanations.

**Outcome:** Provides insights into model decision-making processes, supporting transparency and trust in AI systems.

### 🎬 **Project 4** — Cold-Start Movie Recommender

**Purpose:** Implements a recommender system addressing the cold-start problem using the MovieLens dataset, with PDF export functionality.

**Key Files:**
- `data/{links.csv, movies.csv, ratings.csv, tags.csv}`: MovieLens dataset for recommendations.
- `model/*.npy`: Precomputed matrix factors for collaborative filtering.
- `movie_metadata.json`: Movie metadata.
- `templates/project4/`: Interface for recommendations and PDF generation.
- `static/project4/`: Styling assets.

**Outcome:** Delivers personalized movie recommendations with an intuitive interface and exportable reports.

### 🎮 **Project 5** — Reinforcement Learning with Human Feedback

**Purpose:** Demonstrates reinforcement learning with human feedback (RLHF) using the REINFORCE algorithm to train a cheese-seeking mouse agent.

**Key Files:**
- `models/*.pkl`: Pre-trained RL policies and rewards.
- `utils/`: RL environment and training utilities.
- `templates/project5/`: Interface for policy visualization and feedback.

**Outcome:** Showcases RLHF through an interactive gaming environment, highlighting human-in-the-loop AI refinement.

---

## ⚙️ Setup Instructions

To run the project locally, follow these steps:

### 1. Clone the Repository:
```bash
git clone https://github.com/Sunmeet779/HCAI-PBL.git
cd HCAI-PBL
```

### 2. Create a Virtual Environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 4. Apply Database Migrations:
```bash
python manage.py migrate
```

### 5. Run the Development Server:
```bash
python manage.py runserver
```

Access the application at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

---

## Contributors

**Sunmeet Kohli**  
Matr. Nr: 642365  
Email: Sunmeet.kohli@tuhh.de

**Sanika Acharya**  
Matr. Nr: 640981  
Email: Sanika.acharya@tuhh.de

---

