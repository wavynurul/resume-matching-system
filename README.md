# Resume Matching System

This project is a **Resume Matching System** that leverages **DistilBERT embeddings**, **cosine similarity**, and a modified **Gale-Shapley algorithm** to match job descriptions with resumes effectively.

## Features
- User authentication (Admin & User roles)
- Upload and manage resumes (PDF format)
- Admin can upload job data (CSV)
- Resume and job matching using embeddings
- Evaluation metrics: Mean Average Precision (MAP)
- Visualization of AP score distributions

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **NLP**: Hugging Face Transformers (DistilBERT)
- **ML/DL**: PyTorch
- **Visualization**: Matplotlib, Seaborn

## Dataset

This project uses the **Job Postings Dataset** from Kaggle:  
[Job Postings Dataset on Kaggle](https://www.kaggle.com/datasets/ff0e38aa4a2a813ab9bdb6107b8acaee96407a40eb1362a88380edc49a0a027f)

The dataset includes job descriptions and related information, which were used to train and evaluate the candidate–job matching system.  

For reproducibility, you can download the dataset directly from Kaggle and place it in the `dataset/` folder before running the project.


## 📂 Project Structure

```
ai-job-matching-system/
│
├── dataset/                # Datasets for training and testing
│   ├── job_data.csv        # Folder containing Job postings dataset
│   └── resume/             # Folder containing resumes
│
├── notebooks/              # Jupyter notebooks for training
│
├── src/                    # Source code for the project
│   ├── create_admin_user.py  # Script to create admin user
│   ├── db.py                 # Database connection and operations
│   ├── get_user.py           # Retrieve user data
│   ├── streamlit_app.py      # Streamlit frontend application
│   ├── test_app.py           # Unit tests for the app
│   ├── test_sqlite.py        # Test scripts for SQLite database
│   ├── update_password.py    # Script to update user password
│   └── update_schema.py      # Script to update database schema
│
├── app.py                  # Main entry point for running the application
├── requirements.txt        # Python dependencies
├── .gitignore              # Files and folders to ignore in Git
├── LICENSE                 # MIT License file
└── README.md               # Project documentation
```

---
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wavynurul/resume-matching-system.git
   cd resume-matching-system
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate    # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501/`.

## Notes
- Make sure you have NLTK stopwords and punkt tokenizer downloaded (the app will handle this automatically).
- Ensure job data is uploaded in **CSV format** by the Admin before matching resumes.

## License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
