import streamlit as st
import spacy
from tools import (
    parse_email,
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
# Streamlit UI
st.set_page_config(page_title="Email Automation Agent", layout="centered")
st.title("Email Automation Agent")

# Input fields
subject = st.text_input("Email Subject")
body = st.text_area("Email Body")

if st.button("Submit"):
    if not subject or not body:
        st.warning("Both subject and body are required.")
    else:
        email_input = {
            "subject": subject,
            "body": body
        }

        # Email preprocessing
        cleaned_email_text = parse_email(email_input)

        # Predict email type
        email_type = load_model_and_predict_email_type(cleaned_email_text)

        # Map email type to label
        mapping = {
            'issue': 0,
            'inquiry': 1,
            'suggestion': 2,
            'other matter': 3
        }

        # Predict email criticality
        criticality = load_model_and_predict_criticality(cleaned_email_text, mapping[email_type])

        # Determine routing
        assigned_to = route_email(email_type, criticality)

        # Extract keywords and entities
        keywords, entities = extract_keywords_and_entities(cleaned_email_text)

        # Simulate external task creation
        task_data, api_response = simulate_clickup_task_creation(subject, body, assigned_to)

        automated_reply = generate_automated_reply(email_type, criticality)

        # Display results inside an expander for neatness
        with st.expander("📋", expanded=True):
            st.markdown("### Email Classification")
            st.markdown(f"**Email is classified as :**  `{email_type.capitalize()}`")
            st.markdown(f"**Criticality of email :**  `{criticality.capitalize()}`")


            st.markdown("### Routing Decision")
            st.markdown(f"**Email has been assigned To:**  `{assigned_to}`")

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
        
