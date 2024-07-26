import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from PIL import Image
import pandas as pd
from transformers import pipeline
import torch
import language_tool_python
import matplotlib.pyplot as plt

# Function to load CSV data
@st.cache_data
def load_data(filename):
    return pd.read_csv(filename)

# Determine if a GPU is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1

# Load the pre-trained sentiment analysis model with specified model name and revision
model_name = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
model_revision = 'af0f99b'
sentiment_analyzer = pipeline('sentiment-analysis', model=model_name, revision=model_revision, device=device)

# Function to perform grammar check using language_tool_python library
def perform_grammar_check(text):
    tool = language_tool_python.LanguageTool('en-GB')
    matches = tool.check(text)
    error_count = len(matches)
    
    # Convert error count to a score from 0 to 5
    if error_count == 0:
        return 5
    elif error_count <= 2:
        return 4
    elif error_count <= 5:
        return 3
    elif error_count <= 10:
        return 2
    elif error_count <= 20:
        return 1
    else:
        return 0

# Function to evaluate tone using Hugging Face transformers
def evaluate_tone(customer_msgs, csa_msgs):
    customer_sentiment = sentiment_analyzer(customer_msgs)
    csa_sentiment = sentiment_analyzer(csa_msgs)
    
    customer_label = customer_sentiment[0]['label']
    csa_label = csa_sentiment[0]['label']
    
    if customer_label == 'NEGATIVE' and csa_label in ['POSITIVE', 'NEUTRAL']:
        return 5
    elif customer_label == 'NEGATIVE' and csa_label == 'NEGATIVE':
        return 0
    else:
        return 3

# Function to process the conversation transcript
def process_transcript(transcript, csa_name):
    messages = transcript.split('\n\n')
    customer_msgs = []
    csa_msgs = []

    for message in messages:
        if "Customer:" in message:
            customer_msgs.append(message)
        elif csa_name in message:
            csa_msgs.append(message)
    
    combined_customer_msgs = " ".join(customer_msgs)
    combined_csa_msgs = " ".join(csa_msgs)
    
    grammar_score = perform_grammar_check(combined_csa_msgs)
    tone_score = evaluate_tone(combined_customer_msgs, combined_csa_msgs)
    
    return grammar_score, tone_score

# Function to create PNG files for each CSA with aggregated results
def create_aggregated_results_png(results_df):
    for csa_name, group in results_df.groupby('CSA Name'):
        fig, ax = plt.subplots()
        
        grammar_scores = group['Grammar']
        tone_scores = group['Tone of Voice']
        
        ax.plot(grammar_scores, label='Grammar Score', marker='o')
        ax.plot(tone_scores, label='Tone of Voice Score', marker='x')
        
        ax.set_title(f'Aggregated Results for {csa_name}')
        ax.set_xlabel('Ticket ID')
        ax.set_ylabel('Scores')
        ax.legend()
        
        plt.xticks(rotation=90)
        
        filename = f'{csa_name}_results.png'
        plt.savefig(filename)
        plt.close()

# Streamlit app to upload CSV and display the report
logo_path = "image.png"  # Update with your logo path
logo = Image.open(logo_path)
st.image(logo, width=300)
st.title('COPS: Auto QA Tool - NLP')
uploaded_file = st.file_uploader('Upload CSV file', type='csv')

if uploaded_file is not None:
    data = load_data(uploaded_file)
    results = []

    for index, row in data.iterrows():
        transcript = row['conversation_text']
        csa_name = row['CSA']
        ticket_id = row['Ticket ID']
        
        grammar_score, tone_score = process_transcript(transcript, csa_name)
        
        ticket_url = f"https://app.intercom.com/a/inbox/vyuax3gj/inbox/conversation/{ticket_id}"
        
        results.append({
            'Ticket ID': ticket_url,
            'CSA Name': csa_name,
            'Grammar': grammar_score,
            'Tone of Voice': tone_score
        })
    
    results_df = pd.DataFrame(results)
    st.write(results_df)

    # Append results to NLP_Dump.csv
    dump_file = 'NLP_Dump.csv'
    existing_data = pd.read_csv(dump_file) if os.path.exists(dump_file) else None
    combined_df = pd.concat([existing_data, results_df], ignore_index=True)
    combined_df.to_csv(dump_file, index=False)
    st.success(f"Evaluations have been saved to {dump_file}")

    # Create aggregated results PNG for each CSA
    create_aggregated_results_png(results_df)
    st.success("Aggregated results PNG files have been created for each CSA.")
