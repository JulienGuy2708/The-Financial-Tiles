"""This version of the chatbot uses Mistral 7B instead of BERT to generate answers - 10.04.2024"""


from flask import *
import pandas as pd
from transformers import pipeline
import sys
import os
import tempfile
from io import StringIO
from werkzeug.utils import secure_filename
from tkinter import *
import re
from datetime import datetime
import urllib.request
import torch
import nltk
import pymed
from pymed import PubMed
import json
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.utils import Secret
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder

# Our huggingface token for APIs:
huggingface_token='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'

# generate a secret key:
import secrets
generated_key = secrets.token_hex(16)

app = Flask(__name__)
app.secret_key = generated_key  # Set a secret key for session security

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('data/data.csv')

# Instantiate a question-answering pipeline using a pre-trained model
#qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
#qa_pipeline = pipeline("text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")
# but why not

# instantiate the llm component of the haystack RAG pipeline
llm = HuggingFaceTGIGenerator("mistralai/Mistral-7B-v0.1", token=Secret.from_token(huggingface_token))
llm.warm_up()

# Generate prompt template. Change later following Haystack architecture. 
prompt_template = """
Answer the question truthfully based on the given text.
If the documents don't contain an answer, use your existing knowledge base.
q: {{ question }}
"""
# Initiate the prompt builder
prompt_builder = PromptBuilder(template=prompt_template)

# Create and connect pipeline components
pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.connect("prompt_builder.prompt", "llm.prompt")


@app.route('/')
def home():
    date = datetime.now()
    date_now = date.strftime("%Y%M")

    if 'messages' not in session:
        session['messages'] = []

    return render_template('index.html', messages=session['messages'])

@app.route('/ask', methods=['POST'])
def ask():
    if request.method == 'POST':
        user_question = request.form['user_question']
        #abstracts = df['abstract'].tolist()         # changed Abstract to abstract, now it works
        #abstracts = df['abstract'].tolist()
        abstracts= ""
        
        # Run a Pubmed query based on user input 
        best_answer = find_best_answer(user_question, abstracts)    
        
        # Define the inputs for the prompt builder (this will be changed to a proper prompt builder later)
        question=f"Answer the question truthfully based on the following text or using your existing knowledge.{abstracts}. question: {user_question}"

        # Run the pipeline to answer the question
        answer=pipe.run(data={"prompt_builder":{"question": question},"llm":{"generation_kwargs": {"max_new_tokens": 500}}})

        # Retrieve existing messages from session
        messages = session.get('messages', [])

        # Append user's question and AI's answer to messages
        messages.append(('user', user_question))
        #messages.append(('ai', answer['llm']['replies'][0]))   # take the first answer only to avoid repetitions
        messages.append(('ai', answer['llm']['replies'][0].split('.')[0] + '.'))   # take the first answer only to avoid repetitions
        
        # Update session with the new messages
        session['messages'] = messages

        return render_template('index.html', messages=session['messages'])


def find_best_answer(user_question, abstracts): # replace question by user_question
    # Placeholder logic for finding the most relevant answer. 
    # return abstracts[1]  
    
    # Proposal: let's grab keywords from the user input. As a first approximation for this, extracting the nouns would do. We then use these nouns as keywords      # for automated searches in pubmed using PyMed. We pick up the top 10 or so papers and feed them to M7B. 

    # reset abstracts
    abstracts=""
    
    # Tokenize the user question and keep only nouns
    tokenized = nltk.word_tokenize(user_question) # changed user_question to question
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

    # concatenate the nouns into a single list for use as PyMed query
    nouns = ', '.join(nouns)

    # Run PyMed query with nouns as keywords, keep the top 5 to 10 results (consider expanding later)
    pubmed = PubMed(tool="NiceApp", email="julien.guy@link-intelligence.de")
    query = f'(review[Publication Type]) AND ({nouns}[Text Word])' # restricts article type to reviews for hopefully better search results
    results = pubmed.query(query, max_results=10)

    # concatenate abstracts in a list
    results=list(results)
    for x in range (len(results)):
        abstract_text=str(results[x].abstract)
        abstracts=abstracts+abstract_text

    #Proofread query and results - optional
    print(nouns)
    print(query)
    print(abstracts)
    
    return abstracts

def is_noun(pos):      # This is necessary to extract nouns from user input
    return pos[:2] == 'NN'

    

if __name__ == '__main__':
    app.run(debug=True)

################################################################
# Currently use an in-memory session storage,
# if the Flask server restarts,
# the session data will be lost.
# For a more persistent solution,
# consider using a database to store conversation history