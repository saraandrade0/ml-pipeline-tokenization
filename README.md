
# 🧩 ML Pipeline: Tokenization & Classification

This project demonstrates an **end-to-end machine learning pipeline** for text classification using **Python** and **scikit-learn**.  
It includes everything from preprocessing raw text to training a model and scoring new data.

---

## 🚀 Features
- 📂 Load datasets from CSV (`id`, `text`, `label`)
- ✨ Text preprocessing with **TF-IDF vectorization**
- 🤖 Model training with **Logistic Regression**
- 💾 Save and reuse vectorizers and models with **Joblib**
- 📊 Score new datasets and export results to CSV

---

## 📂 Project Structure
ml-pipeline-tokenization/
├── data/ # sample data (tiny CSVs, not sensitive!)
├── models/ # trained models (ignored in .gitignore)
├── src/
│ ├── tokenization.py # build TF-IDF vocabulary
│ ├── training.py # train logistic regression classifier
│ └── score.py # apply model and generate scores
├── utils.py # helper functions
├── requirements.txt # dependencies
└── README.md


## ⚡ Quickstart

Clone the repository:
```bash
git clone git@github.com:saraandrade0/ml-pipeline-tokenization.git
cd ml-pipeline-tokenization
Create and activate a virtual environment:

Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

Install dependencies:
pip install -r requirements.txt

1️⃣ Build the TF-IDF vectorizer

python src/tokenization.py

2️⃣ Train the classifier

python src/training.py

3️⃣ Score a dataset

python src/score.py data/samples.csv

Output:

models/tfidf.joblib

models/modelo_lr.joblib

scores.csv

📊 Example Results
id	score
1	0.36524
2	0.68042
3	0.22093

🛠️ Tech Stack
Python 3.11+

pandas

scikit-learn

joblib

🔗 Author
Made by Sara Andrade
Feel free to fork, star ⭐, and reach out if you’d like to collaborate!