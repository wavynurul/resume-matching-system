import bcrypt
import sqlite3
from contextlib import closing

def get_user(username, password):
    conn = sqlite3.connect('database.db')
    with closing(conn.cursor()) as cursor:
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        print(f"User fetched: {user}")  # Debug print
        if user and bcrypt.checkpw(password.encode('utf-8'), user[2]):
            return user
        return None

from your_module import get_user

# Test credentials
username = 'admin'
password = 'adminpassword'

user = get_user(username, password)
if user:
    print("Login successful!")
else:
    print("Login failed.")
