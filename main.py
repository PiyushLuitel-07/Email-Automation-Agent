import streamlit as st
import spacy
import re
from tools import (
    parse_email_inputemailisdict_,
    parse_email_inputemailis_emailbytes,
    load_model_and_predict_email_type,
    preprocess,
    load_model_and_predict_criticality,
    route_email,
    simulate_clickup_task_creation,
    generate_automated_reply,
    extract_keywords_and_entities
)
import nltk
from nltk.corpus import stopwords
import json

# Ensure stopwords are available
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# Streamlit UI setup
st.set_page_config(page_title="Email Automation Agent", layout="centered")
st.title("Email Automation Agent")

# Select parsing mode
parsing_option = st.selectbox(
    "Choose input type for email parsing:",
    ("Structured Input (Subject & Body)", "Raw Email Bytes")
)

# Initialize default variables
subject = ""
body = ""
cleaned_email_text = ""

# Option 1: Structured subject & body input
if parsing_option == "Structured Input (Subject & Body)":
    subject = st.text_input("Email Subject")
    body = st.text_area("Email Body")

    if st.button("Submit"):
        if not subject or not body:
            st.warning("Both subject and body are required.")
            st.stop()
        email_input = {"subject": subject, "body": body}
        cleaned_email_text = parse_email_inputemailisdict_(email_input)

# Option 2: Raw email bytes input
elif parsing_option == "Raw Email Bytes":
    raw_email_bytes_input = st.text_area("Paste Raw Email Bytes (as string)")

    if st.button("Submit"):
        if not raw_email_bytes_input:
            st.warning("Raw email bytes input is required.")
            st.stop()
        try:
            raw_bytes = bytes(raw_email_bytes_input, encoding='utf-8')
            subject, body = parse_email_inputemailis_emailbytes(raw_bytes)
            # subject = "Parsed Subject"
            cleaned_email_text = preprocess(subject + " " + body)
            # body = cleaned_email_text  # fallback for task creation
        except Exception as e:
            st.error(f"Error parsing raw email bytes: {e}")
            st.stop()

# Continue if cleaned_email_text is available
if cleaned_email_text:
    # Predict email type
    email_type = load_model_and_predict_email_type(cleaned_email_text)

    # Email type mapping
    mapping = {
        'issue': 0,
        'inquiry': 1,
        'suggestion': 2,
        'other matter': 3
    }

    # Predict criticality
    criticality = load_model_and_predict_criticality(cleaned_email_text, mapping[email_type])

    # Routing decision
    assigned_to = route_email(email_type, criticality)

    # Keywords and entities
    keywords, entities = extract_keywords_and_entities(cleaned_email_text)

    # Simulate task creation
    task_data, api_response = simulate_clickup_task_creation(subject, body, assigned_to)

    # Generate automated reply
    automated_reply = generate_automated_reply(email_type, criticality)

    # Display results
    with st.expander("📋", expanded=True):
        st.markdown("### Email Classification")
        st.markdown(f"**Email is classified as:**  `{email_type.capitalize()}`")
        st.markdown(f"**Criticality of email:**  `{criticality.capitalize()}`")

        st.markdown("### Routing Decision")
        st.markdown(f"**Email has been assigned to:**  `{assigned_to}`")

        st.markdown("### Keywords and Entities")
        st.markdown(f"**Keywords:**  `{', '.join(keywords)}`")
        st.markdown("**Entities:**")
        for ent_text, ent_label in entities:
            st.markdown(f"- `{ent_text}`  —  *{ent_label}*")

        st.markdown("### Simulated External Task Creation")
        st.markdown("Task data sent to the external system:")
        formatted_task_data = json.dumps(task_data, indent=4)
        st.code(formatted_task_data, language='json')

        if api_response["status"] == "success":
            st.success(f"Task successfully simulated. Task ID: `{api_response['task_id']}`")
        else:
            st.error("Failed to simulate task creation. Please try again later.")
            st.markdown(f"Error: `{api_response['error']}`")

        st.markdown("### Automated Reply")
        st.markdown(f"**Automated Reply:**  `{automated_reply}`")
