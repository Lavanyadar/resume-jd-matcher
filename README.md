Resume-JD Matcher
Project Description

In today's competitive job market, manually screening resumes against job descriptions is time-consuming and prone to human bias. The Resume-JD Matcher addresses this challenge by using deep learning to automatically assess how well a candidate's resume aligns with a given job description, enabling faster and more consistent shortlisting.

The core idea behind this project is semantic similarity matching using a Siamese neural network architecture. Instead of treating resume-matching as a simple keyword search, the model learns to encode both the resume and the job description into dense vector representations using a shared encoder. The similarity between these two representations is then used to predict how strong a match the candidate is for the role.

Problem Statement

Given a resume and a job description as input, the goal is to predict a match score (or match/no-match label) that reflects how relevant the candidate's experience and skills are to the requirements of the job — going beyond simple keyword overlap to capture deeper semantic meaning.

Dataset

Two datasets from Kaggle were combined to build the training data:

Resume Dataset — contains resume text labeled by job Category (e.g., Data Science, HR, Sales)
Job Description Dataset — contains job postings with Job Title and full Job Description text

Since the two datasets don't come pre-paired, a category-to-job-title keyword mapping was built to link resumes to relevant job descriptions, generating labeled pairs suitable for supervised training.

Approach & Model Architecture

The project follows an iterative, notebook-by-notebook progression — starting from a simple baseline and moving toward more sophisticated deep learning models, mirroring the structure used in earlier projects (IMDB Sentiment Analysis and Quora Question Pairs):

Logistic Regression (Baseline) — Uses TF-IDF vectorization and classical similarity features to establish a benchmark.
Siamese SimpleRNN — A basic recurrent encoder shared across both text inputs.
Siamese LSTM — Captures longer-range dependencies in resume and job description text.
Siamese GRU — A lighter-weight recurrent alternative to LSTM with comparable performance.
Siamese Transformer (Built from Scratch) — A custom Transformer encoder implemented without pretrained weights, allowing full control and understanding of the attention mechanism applied to this matching task.

Each model outputs embeddings for the resume and job description, which are compared (via distance or similarity function) to produce the final match prediction.

Deployment

The best-performing model is deployed via a Streamlit web app, allowing users to:

Paste in a resume and a job description
Instantly receive a match score/result
Use the tool for quick, interactive screening
Tech Stack
Language: Python
Deep Learning: TensorFlow / Keras
Classical ML: Scikit-learn
Data Handling: Pandas, NumPy
Deployment: Streamlit
Key Highlights
End-to-end pipeline: data pairing → preprocessing → model training → evaluation → deployment
Comparison across 5 different modeling approaches (1 baseline + 4 deep learning models)
Custom-built Transformer encoder (no pretrained weights), demonstrating a strong understanding of attention-based architectures
Real-world applicability in HR tech and recruitment automation
