import streamlit as st
import hashlib
import json
from pathlib import Path

USERS_FILE = Path("webapp/auth/users.json")

def load_users():
    if USERS_FILE.exists():
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login():
    users = load_users()
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username] == hash_password(password):
            st.success("Logged in successfully!")
            return username
        else:
            st.error("Invalid username or password")
    return None

def create_account():
    users = load_users()
    username = st.text_input("Choose a username")
    password = st.text_input("Choose a password", type="password")
    confirm_password = st.text_input("Confirm password", type="password")
    if st.button("Create Account"):
        if username in users:
            st.error("Username already exists")
        elif password != confirm_password:
            st.error("Passwords do not match")
        else:
            users[username] = hash_password(password)
            save_users(users)
            st.success("Account created successfully!")
            return username
    return None

def is_authenticated(username):
    return username is not None