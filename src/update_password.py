import bcrypt
import sqlite3

# Rehash the password and update it
def update_password(username, new_password):
    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET password = ? WHERE username = ?', (hashed_password, username))
    conn.commit()
    conn.close()

# Update admin password
update_password('admin', 'adminpassword')
