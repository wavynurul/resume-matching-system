import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
from collections import defaultdict, deque
import fitz  # PyMuPDF
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizerFast, DistilBertModel
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile
import hashlib
import json

# Ensure necessary NLTK data is downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Set up directories
UPLOAD_DIR = 'uploaded_resumes'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

USER_DATA_DIR = 'user_data'
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

# Sample in-memory user database
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(hashed_password, password):
    return hashed_password == hash_password(password)

users_db = {
    "admin": {
        "password": hash_password("adminpass"),
        "role": "admin"
    },
    # Dummy users
    "user1": {
        "password": hash_password("user1pass"),
        "role": "user"
    },
    "user2": {
        "password": hash_password("user2pass"),
        "role": "user"
    },
    "user3": {
        "password": hash_password("user3pass"),
        "role": "user"
    },
    "user4": {
        "password": hash_password("user4pass"),
        "role": "user"
    },
    "user5": {
        "password": hash_password("user5pass"),
        "role": "user"
    }
}
user_sessions = {}

# Initialize DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

@profile
def extract_text_from_pdf(file):
    # Pastikan file adalah objek file Streamlit
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@profile
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = ' '.join([word for word in words if word not in stop_words])
    return filtered_text

@profile
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

@profile
def gale_shapley_modified(resume_preferences, job_preferences, k):
    resume_matches = defaultdict(list)
    job_matches = defaultdict(list)
    free_resumes = deque(resume_preferences.keys())
    job_proposals = defaultdict(list)

    while free_resumes:
        resume = free_resumes.popleft()
        resume_pref_list = resume_preferences[resume]

        for job_id, score, company in resume_pref_list:
            if job_id not in job_proposals[resume]:
                job_proposals[resume].append(job_id)
                job_matches[job_id].append((resume, company, score))

                if len(job_matches[job_id]) > k:
                    worst_resume = min(job_matches[job_id], key=lambda r: next((x[1] for x in resume_preferences[r[0]] if x[0] == job_id), float('inf')), default=None)

                    if worst_resume:
                        if worst_resume in job_matches[job_id]:
                            job_matches[job_id].remove(worst_resume)
                            if (job_id, worst_resume[1], worst_resume[2]) in resume_matches[worst_resume[0]]:
                                try:
                                    resume_matches[worst_resume[0]].remove((job_id, worst_resume[1]))
                                except ValueError:
                                    pass
                            free_resumes.append(worst_resume[0])

                resume_matches[resume].append((job_id, company, score))
                break

    return resume_matches, job_matches

def average_precision_at_k(relevance_list, k):
    relevance_list = np.asarray(relevance_list)[:k]
    relevant = relevance_list.sum()
    if relevant == 0:
        return 0.0
    score = 0.0
    for i in range(k):
        if relevance_list[i] == 1:
            score += (np.sum(relevance_list[:i + 1]) / (i + 1))
    return score / relevant

def calculate_map(resume_preferences, k, threshold=0.8):
    ap_scores = []
    for resume_id, jobs in resume_preferences.items():
        relevance = np.array([1 if score > threshold else 0 for _, score, _ in jobs])
        ap = average_precision_at_k(relevance, k)
        ap_scores.append(ap)
    return np.mean(ap_scores), ap_scores

def save_user_data(username, resume_file, full_name):
    user_data_path = os.path.join(USER_DATA_DIR, f'{username}_data.json')
    if os.path.exists(user_data_path):
        with open(user_data_path, 'r') as f:
            user_data = json.load(f)
    else:
        user_data = {'full_name': full_name, 'resumes': [], 'status': []}

    user_data['resumes'].append(resume_file)
    user_data['status'].append('Pending')
    
    with open(user_data_path, 'w') as f:
        json.dump(user_data, f)


def update_user_status(username, status_list):
    user_data_path = os.path.join(USER_DATA_DIR, f'{username}_data.json')
    if os.path.exists(user_data_path):
        with open(user_data_path, 'r') as f:
            user_data = json.load(f)
        user_data['status'] = status_list
        with open(user_data_path, 'w') as f:
            json.dump(user_data, f)

def delete_resume(username, resume_index):
    user_data_path = os.path.join(USER_DATA_DIR, f'{username}_data.json')
    if os.path.exists(user_data_path):
        with open(user_data_path, 'r') as f:
            user_data = json.load(f)
        if 0 <= resume_index < len(user_data['resumes']):
            resume_file = user_data['resumes'].pop(resume_index)
            # Remove the resume file from the directory
            resume_path = os.path.join(UPLOAD_DIR, resume_file)
            if os.path.exists(resume_path):
                os.remove(resume_path)
            # Update status
            user_data['status'].pop(resume_index)
            # Save the updated user data
            with open(user_data_path, 'w') as f:
                json.dump(user_data, f)
            st.success("Resume deleted successfully!")
        else:
            st.error("Invalid resume index.")

def logout():
    if 'username' in st.session_state:
        st.session_state.pop('username', None)
    if 'role' in st.session_state:
        st.session_state.pop('role', None)
    st.success("You have been logged out.")
    st.experimental_set_query_params()  # Clear query params to simulate page reload

def login_page():
    st.title("Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users_db and verify_password(users_db[username]["password"], password):
            st.session_state['username'] = username
            st.session_state['role'] = users_db[username]["role"]
            st.success(f"Welcome, {username}!")
            st.experimental_set_query_params()  # Clear query params to simulate page reload
        else:
            st.error("Invalid credentials")

def sign_up_page():
    st.title("Sign Up Page")

    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if username in users_db:
            st.error("Username already exists.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            users_db[username] = {
                "password": hash_password(password),
                "role": "user"
            }
            st.success("Sign up successful! Please log in.")

def user_dashboard():
    st.title("User Dashboard")
    st.write("Welcome, User!")

    if 'username' in st.session_state:
        username = st.session_state['username']
        
        # Input full name
        full_name = st.text_input("Enter Your Full Name", key="full_name_input")
        
        # Upload resume
        resume_file = st.file_uploader("Upload Your Resume PDF", type="pdf", key="resume_upload")
        
        if st.button("Save Data", key="save_data_button"):
            if resume_file and full_name:
                resume_path = os.path.join(UPLOAD_DIR, resume_file.name)
                with open(resume_path, "wb") as f:
                    f.write(resume_file.getbuffer())
                save_user_data(username, resume_file.name, full_name)
                st.success("Resume uploaded and data saved!")
            else:
                st.error("Please upload a resume and enter your full name before saving.")
        
        # View history
        user_data_path = os.path.join(USER_DATA_DIR, f'{username}_data.json')
        if os.path.exists(user_data_path):
            with open(user_data_path, 'r') as f:
                user_data = json.load(f)
            st.write("Your Upload History:")
            for index, (resume, status) in enumerate(zip(user_data['resumes'], user_data['status'])):
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(resume)
                with col2:
                    st.write(f"Status: {status}")
                with col3:
                    if st.button("Delete", key=f"delete_{index}"):
                        delete_resume(username, index)
                        st.experimental_set_query_params()  # Clear query params to simulate page reload
                        st.experimental_rerun()  # Refresh the page to update the list
        else:
            st.write("No upload history found.")

        if st.button("Logout"):
            logout()
    else:
        st.write("Please log in to access your dashboard.")


def admin_dashboard():
    st.title("Admin Dashboard")
    st.write("Welcome, Admin!")

    st.sidebar.subheader("Upload Job Data")
    job_data_file = st.sidebar.file_uploader("Upload Job Data CSV", type="csv", key="job_data_upload")

    if job_data_file:
        job_data_df = pd.read_csv(job_data_file)
        st.write("Job Data Preview:", job_data_df.head())
        job_data_df.to_csv("job_data.csv", index=False)
        st.success("Job data uploaded successfully!")
    else:
        job_data_df = pd.read_csv("job_data.csv") if os.path.exists("job_data.csv") else pd.DataFrame()

    st.sidebar.subheader("Uploaded Resumes")
    uploaded_resumes = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    if uploaded_resumes:
        st.write("List of uploaded resumes:")
        for resume in uploaded_resumes:
            st.write(resume)
            st.download_button(label="Download", data=open(os.path.join(UPLOAD_DIR, resume), 'rb'), file_name=resume)
    else:
        st.write("No resumes have been uploaded yet.")

    if not job_data_df.empty:
        st.write("Job Data:", job_data_df)

        # Admin input for k value and number of companies
        st.sidebar.subheader("Matching Settings")
        k = st.sidebar.text_input("Number of Candidates to Select (k)", value="5")
        num_companies = st.sidebar.text_input("Number of Companies to Randomly Match", value="5")

        try:
            k = int(k)
            num_companies = int(num_companies)
        except ValueError:
            st.error("Please enter valid integers for k and the number of companies.")
            return

        # Perform Matching Button
        if st.sidebar.button("Perform Matching"):
            # Load resume embeddings
            resume_files = [f for f in uploaded_resumes if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
            resumes_texts = [preprocess_text(extract_text_from_pdf(open(os.path.join(UPLOAD_DIR, resume), 'rb'))) for resume in resume_files]
            resume_embeddings = [get_embedding(text) for text in resumes_texts]
            resume_embeddings = torch.stack(resume_embeddings).numpy()

            # Load pre-existing job embeddings
            if os.path.exists('job_embeddings.npy'):
                job_embeddings = np.load('job_embeddings.npy')
            else:
                st.error("Job embeddings file not found.")
                return

            # Load resume metadata for full names
            metadata_available = os.path.exists("resume_metadata.csv")
            if metadata_available:
                 resume_metadata_df = pd.read_csv("resume_metadata.csv")  # This file should include columns 'resume_file' and 'full_name'
                 resume_metadata_df.set_index('resume_file', inplace=True)


            # Randomly select the specified number of job positions
            if num_companies > len(job_data_df):
                st.error(f"The number of companies selected ({num_companies}) exceeds the available job positions.")
            else:
                job_data_subset = job_data_df.sample(n=num_companies, random_state=42)
                job_embeddings_subset = job_embeddings[job_data_subset.index]

                def create_mapping(resume_id, job_data, job_embeddings):
                    similarity_scores = cosine_similarity([resume_embeddings[resume_id]], job_embeddings).flatten()
                    job_data['similarity_score'] = similarity_scores
                    job_data['resume_id'] = f'resume_{resume_id}'
                    sorted_jobs = job_data.sort_values(by='similarity_score', ascending=False)
                    return sorted_jobs[['resume_id', 'company', 'jobdescription', 'similarity_score']]

                all_mappings = pd.DataFrame()
                for resume_id in range(len(resume_files)):
                    resume_mapping = create_mapping(resume_id, job_data_subset.copy(), job_embeddings_subset)
                    all_mappings = pd.concat([all_mappings, resume_mapping])

                def create_preference_lists(job_data):
                    job_preferences = defaultdict(list)
                    resume_preferences = defaultdict(list)

                    for _, row in job_data.iterrows():
                        job_id = row['jobdescription']
                        resume_id = row['resume_id']
                        company = row['company']
                        similarity_score = row['similarity_score']
                        resume_preferences[resume_id].append((job_id, similarity_score, company))
                        job_preferences[job_id].append((resume_id, similarity_score, company))

                    for resume in resume_preferences:
                        resume_preferences[resume] = sorted(resume_preferences[resume], key=lambda x: x[1], reverse=True)

                    for job in job_preferences:
                        job_preferences[job] = sorted(job_preferences[job], key=lambda x: x[1], reverse=True)

                    return resume_preferences, job_preferences

                resume_preferences, job_preferences = create_preference_lists(all_mappings)

                resume_matches, job_matches = gale_shapley_modified(resume_preferences, job_preferences, k)

                def display_matches(resume_matches, job_matches, all_mappings):
                    st.write("### Resume and Job Matching")

                    # Display resume matching results in table
                    resume_match_list = []
                    for resume, jobs in resume_matches.items():
                        if metadata_available:
                            full_name = resume_metadata_df.loc.get(resume, 'full_name', 'Unknown')
                        else:
                            full_name = resume  # Use resume file name as identifier if metadata is not available

                        for job_id, company, score in sorted(jobs, key=lambda x: x[2], reverse=True):
                            score = all_mappings[(all_mappings['resume_id'] == resume) & (all_mappings['jobdescription'] == job_id)]['similarity_score'].values[0]
                            resume_match_list.append({
                                'Full Name': full_name,
                                'Resume': resume,
                                'Job Description': job_id,
                                'Company': company,
                                'Similarity Score': f"{score:.4f}"
                            })
                    
                    if resume_match_list:
                        st.write("#### Resume Matching Results")
                        st.dataframe(pd.DataFrame(resume_match_list))
                    else:
                        st.write("No resume matches found.")


                    # Display job matching results in table
                    job_match_list = []
                    for job, resumes in job_matches.items():
                        for resume, company, score in sorted(resumes, key=lambda x: x[2], reverse=True):
                            if metadata_available:
                                full_name = resume_metadata_df.loc.get(resume, 'full_name', 'Unknown')
                            else:
                                full_name = resume  # Use resume file name as identifier if metadata is not available

                            job_match_list.append({
                                'Job Description': job,
                                'Resume': resume,
                                'Full Name': full_name,
                                'Company': company,
                                'Similarity Score': f"{score:.4f}"
                            })

                    if job_match_list:
                        st.write("#### Job Matching Results")
                        st.dataframe(pd.DataFrame(job_match_list))
                    else:
                        st.write("No matching jobs found.")

                display_matches(resume_matches, job_matches, all_mappings)

                # Calculate MAP and display AP scores
                map_score, ap_scores = calculate_map(resume_preferences, k)
                st.write(f"### Mean Average Precision (MAP) Score: {map_score:.4f}")

                # Plot AP scores distribution
                st.write("#### Average Precision (AP) Scores Distribution")
                fig, ax = plt.subplots()
                sns.histplot(ap_scores, bins=10, kde=True, ax=ax)
                ax.set_title("Distribution of AP Scores")
                st.pyplot(fig)

                # Plot AP scores boxplot
                st.write("#### AP Scores Boxplot")
                fig, ax = plt.subplots()
                sns.boxplot(x=ap_scores, ax=ax)
                ax.set_title("Boxplot of AP Scores")
                st.pyplot(fig)
    else:
        st.write("Please upload job data.")

    if st.sidebar.button("Logout"):
        logout()



def main():
    st.set_page_config(page_title="Resume Matching System", layout="wide")
    
    page = st.sidebar.selectbox("Select Page", ["Login", "Sign Up", "User Dashboard", "Admin Dashboard"])

    if page == "Login":
        login_page()
    elif page == "Sign Up":
        sign_up_page()
    elif page == "User Dashboard":
        if 'username' in st.session_state and st.session_state['role'] == 'user':
            user_dashboard()
        else:
            st.warning("You need to log in as a user to access this page.")
    elif page == "Admin Dashboard":
        if 'username' in st.session_state and st.session_state['role'] == 'admin':
            admin_dashboard()
        else:
            st.warning("You need to log in as an admin to access this page.")

if __name__ == "__main__":
    main()

