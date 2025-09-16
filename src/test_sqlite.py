import sqlite3

# Test SQLite connection
try:
    conn = sqlite3.connect(':memory:')  # Creates an in-memory database
    print("SQLite is installed and working.")
except Exception as e:
    print(f"An error occurred: {e}")
