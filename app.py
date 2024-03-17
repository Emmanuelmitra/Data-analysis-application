import streamlit as st
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro
import base64  # Added base64 for encoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# Custom color values
custom_background_color = "#f0f0f0"
custom_primary_color = "#ff5733"  # Orange
custom_secondary_color = "#333333"  # Dark Gray

# Apply styling with custom colors
st.markdown(
    f"""
    <style>
        body {{ background-color: {custom_background_color}; font-family: 'Arial', sans-serif; }}
        h1, .stApp {{ color: {custom_primary_color}; text-align: center; }}
        h2, h3, .stMarkdown {{ color: {custom_secondary_color}; }}
        .stTextInput, .stFileUploader, .stButton, .stTextArea {{ border-color: {custom_primary_color}; }}
        .stTextInput, .stFileUploader, .stButton:hover {{ background-color: {custom_primary_color}; color: white; }}
    </style>
    """, unsafe_allow_html=True
)

def load_and_display_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(('.csv', '.xls', '.xlsx')):
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")

