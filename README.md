# 📧 Email Automation Agent

## 🧠 Project Overview

The **Email Automation Agent** is an AI-powered solution designed to handle and streamline customer support emails. It automates the following tasks:

- Parses, cleans and preprocesses incoming emails.
- Classifies email **type**: e.g., *Inquiry*, *Suggestion*, *Issue*, or *Other*.
- Predicts email **criticality**: *High*, *Medium*, or *Low* urgency.
- **Routes** emails to appropriate teams based on type and urgency using business rules.
- **Simulates task creation** (e.g., in ClickUp) for tracking and resolution.
- **Auto-generates replies** to customer emails based on it's type.
- Extracts **keywords and named entities** for additional insights and analytics.

The goal is to reduce the manual load of customer service teams and improve response time and efficiency.

---

## 📂 Dataset

**Dataset Used**: [Customer Care Emails Dataset (Kaggle)](https://www.kaggle.com/datasets/rtweera/customer-care-emails)

### Description:
This dataset includes multiple features from which, labeled customer care emails with email type and urgency levels are taken. It was preprocessed and used to train two main models:

- **Email Type Classifier**: Categorizes emails into `Inquiry`, `Suggestion`, `Issue`, `Other Matters`.
- **Criticality Predictor**: Estimates how urgent the email is – `High`, `Medium`, or `Low`.

---

## 🧱 Project Structure

```bash
├── tools.py                # Core logic functions
├── main.py                 # Streamlit-based web application
├── data/                   # Raw and preprocessed datasets
├── models/                 # Saved ML models and vectorizers
├── notebooks/              # Data exploration and model training notebooks
├── README.md               # Project documentation (this file)
```

---

## ⚙️ Core Functionalities (tools.py)

### 1. `parse_email_inputemailis_emailbytes(raw_email_bytes)`

- **Input**: Raw email data as **bytes** (e.g., from `.eml` files or email servers).
- **Steps**:
  - Use Python’s `email` library to parse subject and body from raw email bytes.
  - Extract plain text body, even if the email is multipart.
  - Define a `clean_text()` helper to:
    - Remove URLs using regex.
    - Normalize whitespace.
    - Remove special characters.
    - Convert to lowercase.
  - Clean both subject and body.
  - Return both cleaned subject and body.

- **Output**: A tuple `(cleaned_subject, cleaned_body)` as two cleaned strings.

✅ *Why*: This function ensures robust parsing of real-world email formats, supporting raw `.eml` byte content and handling multipart emails correctly.

```python
def parse_email_inputemailis_emailbytes(raw_email_bytes):
    """
    Parses real-life raw email bytes and returns a cleaned string (subject + body).
    
    Args:
        raw_email_bytes (bytes): Raw email data from .eml file or email server.
    
    Returns:
        str: Cleaned and combined subject and body text.
    """
    # Parse the raw email bytes
    msg = BytesParser(policy=policy.default).parsebytes(raw_email_bytes)

    subject = msg['subject'] or ''
    body = ''

    # Extract plain text body
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain' and not part.get_content_disposition():
                body = part.get_content()
                break
    else:
        body = msg.get_content()

    # Cleaning function
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)            # Remove URLs
        text = re.sub(r'\s+', ' ', text)               # Normalize whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)     # Remove special characters
        return text.strip().lower()

    cleaned_subject = clean_text(subject)
    cleaned_body = clean_text(body)

    print(cleaned_subject)
    print(cleaned_body)

    # Combine subject and body for classification
    return cleaned_subject, cleaned_body

```

---

### 2. `parse_email_inputemailisdict_(email_raw)`

- **Input**: Email as a **dictionary** with `'subject'` and `'body'` keys.
- **Steps**:
  - Extract `subject` and `body` from dictionary keys.
  - Use a `clean_text()` helper to:
    - Remove URLs.
    - Normalize whitespace.
    - Remove special characters.
    - Convert to lowercase.
  - Clean both subject and body text.
  - Concatenate and return the cleaned subject and body as a single string.

- **Output**: A single cleaned string containing both subject and body.

✅ *Why*: Used when structured data is already available (like from a web form or dataset). It prepares a normalized input string for downstream classification and analysis.

```python
def parse_email_inputemailisdict_(email_raw):
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

```
---

### 3. `load_model_and_predict_email_type(cleaned_email_text)`

- **Input**: Cleaned email content.
- **Steps**:
  - Load the trained model (`email_type_model.pkl`) and vectorizer (`email_vectorizer.pkl`).
  - Vectorize the input text using `TfidfVectorizer`.
  - Predict the email type (label 0–3) and return the human-readable type.

- **Model Used**: `RandomForestClassifier` with TF-IDF features.

✅ *Why*: Logistic Regression gives robust classification for textual data and handles small to medium datasets effectively.

```python
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

```
---

### 4. `load_model_and_predict_criticality(cleaned_email_text, email_type_label)`

- **Input**: Cleaned email and predicted email type (as label).
- **Steps**:
  - Load urgency prediction model and vectorizer.
  - Create a feature vector using:
    - Email TF-IDF vector
    - Email type label (added as a feature).
  - Predict urgency level (`High`, `Medium`, `Low`).

- **Model Used**: `Logistic Regression` for interpretable urgency prediction.

✅ *Why*: Logistic Regression works well with added categorical features and gives interpretable results for urgency.
```python
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

```

---

### 5. `route_email(email_type, criticality)`

- **Input**: `email_type` and `criticality`


```python
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

```
✅ *Why*: Rule-based routing ensures reliability, quick decision-making, and custom business logic without needing a learning model.

---

### 6. `simulate_clickup_task_creation(subject, body, assigned_to)`

- **Input**: Email `subject`, `body`, and `assigned_to` team.
- **Steps**:
  - Simulate a ClickUp API payload.
  - Return JSON structure of mock task with success status.

✅ *Why*: Simulates integration with external ticketing or task tracking systems.

```python
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

```
---

### 7. `generate_automated_reply(email_type, criticality)`

- **Input**: `email_type`, `criticality`.


```python
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
```
✅ *Why*: Speeds up initial acknowledgment and improves customer experience.

---

### 8. `extract_keywords_and_entities(cleaned_email_text)`

- **Input**: Cleaned email text.
- **Steps**:
  - Use **spaCy** NLP pipeline.
  - Extract:
    - **Keywords**: Based on noun chunks and root tokens.
    - **Named Entities**: Persons, Orgs, Products, etc.
- **Output**: List of keywords and named entity tuples.

✅ *Why*: Helps in downstream tasks like analytics, filtering, and trend analysis.
```python
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
```

---


## 🧩 Streamlit Workflow (main.py)

1. **Email Input Form**
   - User enters subject and body of the email.

2. **Validation**
   - Ensures fields are not empty.

3. **Parse Email**
   - `parse_email()` → Cleaned text

4. **Classify Email Type**
   - `load_model_and_predict_email_type()` → Type label and readable string

5. **Predict Urgency**
   - `load_model_and_predict_criticality()` → Urgency level

6. **Route Email**
   - `route_email()` → Assigned department

7. **Extract NLP Insights**
   - `extract_keywords_and_entities()` → Keywords and entities

8. **Simulate Task Creation**
   - `simulate_clickup_task_creation()` → Returns task JSON

9. **Generate Auto Reply**
   - `generate_automated_reply()` → Email body for reply

10. **Results Displayed**
   - Email Type, Criticality, Assigned Team
   - Keywords & Entities
   - Simulated Task Info
   - Generated Reply

---

## 🧪 Model Training Strategy

### Preprocessing (done in `notebooks/`):
- Removed HTML tags, punctuations, stopwords.
- Used `TfidfVectorizer` for feature extraction.
- Split dataset using stratified sampling to preserve label balance.

### Email Type Classifier:
- **Model**: `Logistic Regression`
- **Vectorizer**: `TfidfVectorizer`
- **Reason**: Good for high-dimensional sparse data like text.

### Urgency Predictor:
- **Model**: `LogisticRegression`
- **Input**: Combined TF-IDF + email type label
- **Reason**: Suitable for ordinal urgency levels and easy to interpret.

---

## Results

![Dashboard](https://raw.githubusercontent.com/PiyushLuitel-07/Email-Automation-Agent/main/images/1_dashboard.png)
![Image 2](https://raw.githubusercontent.com/PiyushLuitel-07/Email-Automation-Agent/main/images/2.png)
![Image 3](https://raw.githubusercontent.com/PiyushLuitel-07/Email-Automation-Agent/main/images/3_.png)
![Image 4](https://raw.githubusercontent.com/PiyushLuitel-07/Email-Automation-Agent/main/images/4_.png)
![Image 5](https://raw.githubusercontent.com/PiyushLuitel-07/Email-Automation-Agent/main/images/5_.png)
![Image 6](https://raw.githubusercontent.com/PiyushLuitel-07/Email-Automation-Agent/main/images/6_.png)


## ▶️ Running the App

```bash
streamlit run main.py
```

Enter an email subject and body to simulate the complete automation pipeline in real-time.

---

## 🚀 Features Summary

| Feature                     | Description                                               |
|----------------------------|-----------------------------------------------------------|
| Email Parsing              | Cleans and preprocesses subject + body.                   |
| Email Type Classification  | Predicts: Inquiry, Suggestion, Issue, Other.              |
| Urgency Prediction         | Predicts: High, Medium, Low urgency.                      |
| Rule-based Routing         | Routes email to right department.                         |
| Task Simulation            | Mimics creating a ClickUp task.                           |
| Auto-reply Generation      | Crafts a predefined reply based on type and urgency.      |
| Keyword/Entity Extraction  | NLP-based extraction using spaCy.                         |

---

## 📌 Future Enhancements

- Use LLMs (e.g., Gemini Pro or GPT-4) **only when traditional models are uncertain**.
- Real API integration with task managers (ClickUp, Jira).
- Extend to handle attachments or multi-lingual emails.
- Active learning loop for improving classification via human feedback.

---

## 👨‍💻 Author

**Piyush Luitel**  
Email: luitepiyush96@gmail.com  
GitHub: [PiyushLuitel-07](https://github.com/PiyushLuitel-07)  
LinkedIn: [piyushluitel](https://linkedin.com/in/piyushluitel)  
YouTube: [@piyushluitel7](https://youtube.com/@piyushluitel7)
