import streamlit as st
import pickle
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load models
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

job_model = pickle.load(open(os.path.join(BASE_DIR, "job_role_model.pkl"), "rb"))
recruiter_model = pickle.load(open(os.path.join(BASE_DIR, "recruiter_model.pkl"), "rb"))
tfidf = pickle.load(open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb"))

# -------------------------------
# Utility functions
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

def predict_job_role(skills, education, experience):
    text = f"{skills} {education} {experience}"
    text = clean_text(text)
    vec = tfidf.transform([text])
    return job_model.predict(vec)[0]

def predict_recruiter_decision(skills, education, experience):
    text = f"{skills} {education} {experience}"
    text = clean_text(text)
    vec = tfidf.transform([text])
    return recruiter_model.predict(vec)[0]

def job_fit_score(resume_text, job_description):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)
    vecs = tfidf.transform([resume_clean, job_clean])
    score = cosine_similarity(vecs[0], vecs[1])[0][0]
    return round(float(score * 100), 2)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("AI Resume Screening System")

skills = st.text_area("Skills")
education = st.text_input("Education")
experience = st.number_input("Experience (Years)", 0, 50, 0)
job_desc = st.text_area("Job Description")

if st.button("Analyze Resume"):
    job_role = predict_job_role(skills, education, experience)
    decision = predict_recruiter_decision(skills, education, experience)

    resume_text = f"{skills} {education} {experience}"
    fit_score = job_fit_score(resume_text, job_desc)

    st.success(f"Predicted Job Role: {job_role}")
    st.info(f"Recruiter Decision: {decision}")
    st.warning(f"Job Fit Score: {fit_score}%")
