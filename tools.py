import re
import joblib
import nltk
from nltk.corpus import stopwords
import requests
import streamlit as st
import spacy

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def parse_email(email_raw):
    """
    Parses raw email input and returns a cleaned string (subject + body).
    
    Args:
        email_raw (dict): Dictionary with 'subject' and 'body' as keys.
    
    Returns:
        str: Cleaned combined email text.
    """
    subject = email_raw.get('subject', '')
    body = email_raw.get('body', '')

    # Remove special characters, URLs, and excessive whitespace
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)  # remove URLs
        text = re.sub(r'\s+', ' ', text)     # normalize whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special chars
        return text.strip().lower()
    
    cleaned_subject = clean_text(subject)
    cleaned_body = clean_text(body)

    # Combine subject and body for classification
    return f"{cleaned_subject} {cleaned_body}"


def preprocess(text):
    text = text.lower()
    text = text.replace('\n', ' ')  # replace newlines with space
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def load_model_and_predict_email_type(starting_text):
    # Load the model and vectorizer
    model = joblib.load('models/1_email_type_classifier.pkl')
    vectorizer = joblib.load('models/1_tfidf_vectorizer_email_type_classifier.pkl')
    
    # Preprocess the input text
    processed_text = preprocess(starting_text)
    
    # Transform the text using the vectorizer
    text_vec = vectorizer.transform([processed_text])
    
    # Predict the label
    predicted_label = model.predict(text_vec)[0]

    mapping = {
    'issue': 0,
    'inquiry': 1,
    'suggestion': 2,
    'other matter': 3
    }
    
    # Map the label back to email type
    for key, value in mapping.items():
        if value == predicted_label:
            return key


from scipy.sparse import csr_matrix, hstack
#function to load the model and vectorizer and predict the email criticality
def load_model_and_predict_criticality(starting_text, email_type):
    # Load the model and vectorizer
    model = joblib.load('models/2_email_criticality_classifier.pkl')
    tfidf = joblib.load('models/2_tfidf_vectorizer_email_criticality_classifier.pkl')
    
    # Preprocess the input text
    processed_text = preprocess(starting_text)
    
    # Transform the text using the vectorizer
    text_vec = tfidf.transform([processed_text])
    
    # Prepare email type for prediction
    email_type_vec = csr_matrix([[email_type]])
    
    # Combine text vector and email type
    final_vec = hstack([text_vec, email_type_vec])
    
    # Predict the urgency
    predicted_urgency = model.predict(final_vec)[0]

    # Map the urgency back to its original label
    urgency_mapping = {1: 'low', 2: 'medium', 3: 'high'}
    predicted_urgency = urgency_mapping.get(predicted_urgency, 'unknown')
    
    
    return predicted_urgency



def route_email(email_type, criticality):
    """
    Determines the correct team/person to forward the email to based on type and criticality.
    """
    routing_rules = {
        'issue': {
            'high': 'Tech Support Team',
            'medium': 'Customer Care Team',
            'low': 'Customer Care Team'
        },
        'inquiry': {
            'high': 'Sales Lead',
            'medium': 'Info Desk',
            'low': 'Info Desk'
        },
        'suggestion': {
            'high': 'Product Team',
            'medium': 'Product Team',
            'low': 'Feedback Coordinator'
        },
        'other matter': {
            'high': 'Operations Manager',
            'medium': 'General Admin',
            'low': 'General Admin'
        }
    }
    
    return routing_rules.get(email_type, {}).get(criticality.lower(), "General Admin")


def simulate_clickup_task_creation(subject, body, assigned_to):
    """
    Simulates the creation of a task in an external project management tool like ClickUp.
    Returns the simulated task data and response for display outside this function.
    """

    task_data = {
        "Title": subject,
        "Description": body,
        "Assigned To": assigned_to,
        "Status": "Open"
        # "Priority": "High"
    }

    simulated_response = {
        "status": "success",
        "task_id": "TASK-12345"
    }

    return task_data, simulated_response

def generate_automated_reply(email_type, criticality):
    # Basic templates, customize as needed
    replies = {
        "issue": "Thank you for reporting the issue. Our support team will get back to you shortly.",
        "inquiry": "Thank you for your inquiry. We will respond with the information soon.",
        "suggestion": "Thank you for your suggestion! We appreciate your feedback.",
        "other matter": "Thank you for reaching out. We will review your email and get back to you."
    }
    
    # Add note for urgent cases
    if criticality.lower() == "high":
        replies[email_type] += " Your request has been marked urgent and prioritized accordingly."

    return replies.get(email_type, "Thank you for contacting us.")



def extract_keywords_and_entities(text):
    """
    Extract keywords (noun chunks + named entities) and named entities from the concatenated email text.
    Input:
        text (str): concatenated subject and body of email
    Returns:
        keywords (list of str): unique keywords extracted from noun chunks and entities
        entities (list of tuples): extracted named entities as (entity_text, entity_label)
    """
    try:
        doc = nlp(text)
        
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        noun_chunks = set(chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1)
        entity_texts = set(ent[0].lower() for ent in entities)
        
        keywords = list(noun_chunks.union(entity_texts))
        
        return keywords, entities
    
    except Exception as e:
        print(f"Error extracting keywords/entities: {e}")
        return [], []