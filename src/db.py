import sqlite3
from contextlib import closing
import pandas as pd

# Initialize database
def initialize_db():
    with closing(sqlite3.connect('database.db')) as conn:
        with conn:
            # Drop tables if they exist to reset schema
            conn.execute('DROP TABLE IF EXISTS users')
            conn.execute('DROP TABLE IF EXISTS resumes')
            conn.execute('DROP TABLE IF EXISTS matching_results')

            # Create users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user'
                )
            ''')

            # Create resumes table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS resumes (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    full_name TEXT,
                    resume BLOB,
                    uploaded_at TEXT
                )
            ''')

            # Create matching_results table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS matching_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resume_id TEXT NOT NULL,
                    username TEXT NOT NULL,
                    full_name TEXT,
                    job_description TEXT,
                    company TEXT,
                    cosine_score REAL,
                    precision_map TEXT,
                    accepted INTEGER
                )
            ''')

def get_db_connection():
    return sqlite3.connect('database.db')

def save_resume(username, resume_id, full_name, resume):
    with closing(get_db_connection()) as conn:
        with conn:
            try:
                conn.execute('''
                    INSERT INTO resumes (id, username, full_name, resume, uploaded_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (resume_id, username, full_name, sqlite3.Binary(resume), pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")))
            except Exception as e:
                print(f"An error occurred while saving the resume: {e}")
                raise

def fetch_user_resumes(username):
    with closing(get_db_connection()) as conn:
        return conn.execute('''
            SELECT id, username, full_name, resume, uploaded_at FROM resumes WHERE username = ?
        ''', (username,)).fetchall()

def fetch_all_resumes():
    with closing(get_db_connection()) as conn:
        return conn.execute('''
            SELECT id, username, full_name, resume, uploaded_at FROM resumes
        ''').fetchall()

def save_job_data(job_description):
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute('''
                INSERT INTO job_data (job_description)
                VALUES (?)
            ''', (job_description,))

def save_matching_results(resume_id, username, full_name, job_description, company, cosine_score, precision_map, accepted):
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute('''
                INSERT INTO matching_results (resume_id, username, full_name, job_description, company, cosine_score, precision_map, accepted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (resume_id, username, full_name, job_description, company, cosine_score, precision_map, accepted))

def fetch_matching_results():
    with closing(get_db_connection()) as conn:
        return conn.execute('''
            SELECT * FROM matching_results
        ''').fetchall()

def get_user(username, password):
    with closing(get_db_connection()) as conn:
        return conn.execute('''
            SELECT * FROM users WHERE username = ? AND password = ?
        ''', (username, password)).fetchone()


def create_user(username, password, role='user'):
    with closing(get_db_connection()) as conn:
        with conn:
            conn.execute('''
                INSERT INTO users (username, password, role)
                VALUES (?, ?, ?)
            ''', (username, password, role))
