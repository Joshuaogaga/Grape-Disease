import streamlit as st
from cnn_page import cnn_classifier_page
from vit_page import vit_classifier_page

# Set the page configuration for the main app
st.set_page_config(page_title="Grape Disease Classifier", layout="centered")

st.title("🌿 Grape Disease Classifier - Multi-version App")

# Add a sidebar to switch between the CNN and ViT versions
page = st.sidebar.selectbox("Select Classifier Version", ["CNN Classifier", "ViT Classifier"])

if page == "CNN Classifier":
    cnn_classifier_page()  # Loads CNN version
elif page == "ViT Classifier":
    vit_classifier_page()  # Loads ViT version