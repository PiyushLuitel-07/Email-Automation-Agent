# Email Automation Agent

## Project Overview

This project automates the handling of incoming customer support emails by building an AI-powered email agent that:

- Parses incoming emails to clean and prepare the text.
- Classifies emails into types like inquiry, suggestion, issue or other matters.
- Predicts the urgency or criticality level of each email.
- Routes the email to the appropriate team based on type and urgency.
- Simulates creating a task in a project management tool (like ClickUp).
- Generates automated replies to customers.
- Extracts keywords and entities for further insights.

This reduces manual workload for the support team and speeds up responses.

---

## Dataset Used

We used the publicly available **Customer Care Emails Dataset** from Kaggle:

- URL: https://www.kaggle.com/datasets/rtweera/customer-care-emails

This dataset contains several columns which after cleaning now contains labeled emails with categories and urgency labels. It helped train:

- A model to classify email types.
- A model to predict email urgency/criticality.

---

## Project Structure

- **tools.py**: Contains all core functions performing email parsing, prediction, routing, task simulation, reply generation, and NLP extraction.
- **main.py**: Streamlit web app that takes email input and calls the functions in tools.py to run the full automation pipeline.
- **data/**: Contains the raw and processed dataset for classification
- **notebooks/**: Contains the necessary jupyter notebooks for preprocessing and testing
- **models/**: Contains the classification models

---

## Detailed Explanation of Functions in tools.py

### 1. `parse_email(email_dict)`

- **Input:** A dictionary with keys `subject` and `body` containing raw email content.
- **Process:**
  - Combine subject and body into one string.
  - Remove punctuation, convert text to lowercase.
  - Remove common stopwords (like “the”, “is”, etc.).
  - Tokenize and rejoin cleaned words into a normalized string.
- **Output:** Cleaned, preprocessed email text.

*Why?*  
Cleaning text removes noise and ensures the model sees standardized input for better classification.

---

### 2. `load_model_and_predict_email_type(cleaned_email_text)`

- **Input:** Preprocessed email text.
- **Process:**
  - Load a saved machine learning model and text vectorizer from disk.
  - Vectorize the input text using the same vectorizer.
  - Predict the email type label: inquiry, suggestion, issue or other matters.
- **Output:** Email type as a string label.

*Why?*  
Classifying the email type allows routing to the correct support team.

---

### 3. `load_model_and_predict_criticality(cleaned_email_text, email_type_label)`

- **Input:** Cleaned email text and the numeric label for email type.
- **Process:**
  - Load a different trained model for urgency prediction.
  - Use the email text features and type label as inputs.
  - Predict the criticality level (e.g., high, medium, low urgency).
- **Output:** Urgency/criticality label as string.

*Why?*  
Helps prioritize urgent emails that require faster handling.

---

### 4. `route_email(email_type, criticality)`

- **Input:** Email type and urgency.
- **Process:**  
Use hard-coded business rules:
  - Complaints and high-urgency → Customer Support Team.
  - Technical issues → Engineering Team.
  - Orders → Sales Team.
  - Others → General Support.
- **Output:** Name of team or person assigned.

*Why?*  
Routing ensures emails reach the right experts for efficient resolution.

---

### 5. `simulate_clickup_task_creation(subject, body, assigned_to)`

- **Input:** Email subject, body, and assignee.
- **Process:**  
Simulate an API call to ClickUp or any task management tool by:
  - Creating a JSON payload with task details.
  - Mocking a successful API response.
- **Output:** The task data and fake API response.

*Why?*  
Demonstrates integration with external workflow tools, automating ticket creation.

---

### 6. `generate_automated_reply(email_type, criticality)`

- **Input:** Email type and urgency.
- **Process:**  
Use predefined response templates tailored for email types and urgency.
- **Output:** Auto-generated reply text.

*Why?*  
Sends prompt acknowledgment to customers without manual typing.

---

### 7. `extract_keywords_and_entities(cleaned_email_text)`

- **Input:** Cleaned email text.
- **Process:**  
Use NLP libraries (like spaCy) to:
  - Identify important keywords.
  - Extract named entities (people, organizations, locations).
- **Output:** List of keywords and a list of tuples (entity text, entity type).

*Why?*  
Adds context and can be used for analytics or improving routing.

---

## Workflow in main.py (Step-by-Step)

1. **User Input:**  
   Via Streamlit UI, the user inputs email subject and body.

2. **Validation:**  
   Checks if both subject and body are provided.

3. **Parse Email:**  
   Calls `parse_email()` to clean the raw email content.

4. **Predict Email Type:**  
   Calls `load_model_and_predict_email_type()` with cleaned text.

5. **Predict Criticality:**  
   Calls `load_model_and_predict_criticality()` using cleaned text and email type.

6. **Routing Decision:**  
   Calls `route_email()` to assign the email to the correct team.

7. **Extract Keywords and Entities:**  
   Calls `extract_keywords_and_entities()` for further insights.

8. **Simulate Task Creation:**  
   Calls `simulate_clickup_task_creation()` to mimic opening a ticket.

9. **Generate Reply:**  
   Calls `generate_automated_reply()` to prepare an auto-response.

10. **Display Results:**  
    Shows all outputs (classification, routing, task info, reply, keywords) in the web app.

---

## How to Use

- Run the app with:
  ```bash
  streamlit run main.py
