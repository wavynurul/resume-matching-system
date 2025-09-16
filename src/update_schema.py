import sqlite3
from contextlib import closing
import bcrypt

def update_schema():
    # Connect to the database
    with closing(sqlite3.connect('database.db')) as conn:
        with closing(conn.cursor()) as cursor:
            # Check if the 'role' column exists
            cursor.execute("PRAGMA table_info(users)")
            columns = [column[1] for column in cursor.fetchall()]

            if 'role' not in columns:
                try:
                    cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
                    conn.commit()
                    print("Database schema updated successfully.")
                except sqlite3.OperationalError as e:
                    print(f"Error updating schema: {e}")
            else:
                print("Column 'role' already exists. No changes made.")

            # Update existing data
            cursor.execute('UPDATE users SET role = ? WHERE username = ?', ('admin', 'admin'))
            conn.commit()

            # Example: Insert new admin user with hashed password
            hashed_password = bcrypt.hashpw('adminpassword'.encode('utf-8'), bcrypt.gensalt())
            cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', ('admin', hashed_password, 'admin'))
            conn.commit()
            print("Data updated successfully!")

if __name__ == "__main__":
    update_schema()
