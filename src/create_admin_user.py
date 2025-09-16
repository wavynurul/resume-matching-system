import bcrypt
import sqlite3

# Original plain-text password
password = 'adminpassword'

# Hash the password
hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Connect to the database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Delete existing admin user
cursor.execute('DELETE FROM users WHERE username = ?', ('admin',))
print("Existing admin user deleted (if it existed).")

# Insert new admin user
try:
    cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', ('admin', hashed_password, 'admin'))
    print("Admin user created successfully.")
except sqlite3.IntegrityError:
    print("Admin user already exists. No changes made.")

conn.commit()
conn.close()
