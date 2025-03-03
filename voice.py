import os
from urllib.parse import quote
import speech_recognition as sr
import streamlit as st
import mysql.connector
import requests
import streamlit.components.v1 as components
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import GoogleGenerativeAI
import threading
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
import validators
import re
from langdetect import detect
from googletrans import Translator
import langid
from langdetect import detect, LangDetectException

translator = Translator()

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Database connection parameters
user = "root"
password = "Sensegu@01"
host = "localhost"
port = 3306
database = "store"

# Create SQL
encoded_password = quote(password)
db_uri = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{database}"

# Initialize SQLDatabase
db = SQLDatabase.from_uri(db_uri)

# Initialize SentenceTransformer model for embedding queries
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize LLM (GoogleGenerativeAI) for query generation
try:
    llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

# Prepare the few-shot prompt examples for dynamic SQL generation
few_shot_prompt = """Given the following user question, generate a clean and executable SQL query to fetch the relevant product data from the database. Always ensure that the product name (or a similar attribute) is part of the query, even if the question doesn't directly specify it.

Examples:

Question: What is the price of the Logitech Z623 speakers?
SQL Query: SELECT product_name, product_description, product_price, FROM products WHERE product_name LIKE '%Logitech Z623%' LIMIT 5;

Question: What is the discount on the Dell XPS 13 laptop?
SQL Query: SELECT product_name, product_description, product_price, product_quantity, discount_percent FROM products WHERE product_name LIKE '%Dell XPS 13%' LIMIT 5;

Question: Show me sony headphone
SQL Query: SELECT product_name, product_description, product_price, FROM products WHERE product_name LIKE '%Sony%' and product_type = '%Headphones%' LIMIT 5;

Question: Show me audio devices
SQL Query: SELECT product_name, product_description, product_price, FROM products WHERE (product_type = '%audio%' OR product_category = '%audio%') LIMIT 5;

Question: Get the products that are on sale.
SQL Query: SELECT product_name, product_description, product_price, product_quantity, discount_percent FROM products WHERE discount_percent > 0 AND discount_percent IS NOT NULL LIMIT 5;

Question: Show me the products with no discount.
SQL Query: SELECT product_name, product_description, product_price, FROM products WHERE discount_percent = 0 OR discount_percent IS NULL LIMIT 5;

Question: Show me the products under the 'Smartphone' product type.
SQL Query: SELECT product_name, product_description, product_price, FROM products WHERE product_type = 'Smartphone' LIMIT 5;

Question: Do you sell pet food?
SQL Query: SELECT product_name, product_description, product_price, FROM products WHERE product_category LIKE %Pet Food% LIMIT 5;

Question: 300円以下のジャケットが必要です
SQL Query: SELECT product_name, product_description, product_price, FROM products WHERE product_type LIKE %Jacket% LIMIT 5;

Question: {question}
SQL Query:  # Only return the SQL query, with no extra explanations or formatting
"""

detected_language = "ja"  # Example: Japanese
answer_prompt = PromptTemplate.from_template(
    f"""Given the following user question, SQL query, and query result, provide a human-readable answer in {detected_language} in the form of a list of bullet points.

Question: {{question}}
SQL Query: {{query}}
SQL Result: {{result}}

Your response should be formatted as follows:
- Point 1
- Point 2
- Point 3

Make sure the response starts directly with the bullet points, without any additional labels like "Answer:" or headers."""
)

# Define the PromptTemplate for generating SQL queries based on user questions
prompt_template = PromptTemplate(
    input_variables=["question"],  # We're using the question from the user
    template=few_shot_prompt + "\nQuestion: {question}\nSQL Query:"  # Concatenate few-shot examples with the user's question
)

# Set up the LLM chain for generating SQL queries
chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to detect language of user input
def detect_language(text):
    try:
        # Clean up the text by removing unwanted characters (non-alphanumeric)
        cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        
        if not cleaned_text:
            return "en"  # Default to English if the cleaned text is empty
        
        # Try to detect language using langdetect
        lang = detect(cleaned_text)  
                
        # Confidence filtering based on length of input or misdetected languages
        if len(cleaned_text.split()) < 3:  # Short text handling
            return "en"  # Default to English for short inputs
        
        # Additional check for common misdetections
        if lang not in ['ja', 'es', 'fr', 'de', 'en', 'zh-cn', 'zh-tw']:
            return "en"

        return lang
    except LangDetectException as e:
        # If langdetect fails, use langid as fallback
        try:
            lang, confidence = langid.classify(text)  # langid also returns a confidence score

            # If the confidence is too low (below 0.7), we can assume it may be incorrect
            if confidence < 0.7:
                return "en"  # Default to English if confidence is too low
            return lang
        except Exception as e:
            return "en"  # Default to English if both methods fail
        
# Function to translate text to English or detected language
def translate_text(text, src_lang):
    try:
        # If the language is not English, we will translate to English
        if src_lang != 'en':
            translated = translator.translate(text, src=src_lang, dest='en')  # Translate to English
            return translated.text
        else:
            # If the language is already English, no need to translate
            return text
    except Exception as e:
        return text  # Return original text if translation fails
    
def clean_sql_query(query):
    query = query.strip()  # Remove leading/trailing whitespace
    query = re.sub(r'^```sql\s*', '', query)  # Remove leading ```sql if present
    query = re.sub(r'```$', '', query)  # Remove trailing ``` if present
    return query

def execute_query(query):
    try:
        result_query = chain.run(question=query)  # This gets the SQL query from the LLM
        print(f"Generated SQL Query: {result_query}")  # Log the generated query

        # Clean the SQL query to remove extra text or formatting
        cleaned_query = clean_sql_query(result_query)
        print(f"Cleaned SQL Query: {cleaned_query}")  # Log the cleaned query

        # Ensure only one query is passed to execute
        cleaned_query = cleaned_query.strip()  # Remove any leading/trailing whitespaces

        # Execute the query in the MySQL database
        connection = mysql.connector.connect(user=user, password=password, host=host, port=port, database=database)
        cursor = connection.cursor(dictionary=True)
        cursor.execute(cleaned_query)  # Execute the cleaned query
        
        # Fetch the results of the query
        result = cursor.fetchall()
        print(f"Query result: {result}")  # Log the result

        cursor.close()
        connection.close()

        # Prepare the result data to be used in the template
        result_data = []
        for row in result:
            product_info = {
                "product_name": row["product_name"],
                "product_description": row.get("product_description", ""),
                "product_price": f"¥{float(row.get('product_price', 0.0)):.2f}" if row.get('product_price') is not None else "N/A",
                "discount_percent": row.get("discount_percent", 0),
            }
            result_data.append(product_info)

        # Generate the answer based on the SQL result
        prompt_input = {
            "question": query,
            "query": cleaned_query,
            "result": result_data  # Pass the formatted results to the answer generation
        }
        print(f"Prompt Input: {prompt_input}")  # Log the data being passed to the prompt
        # Generate the final answer
        answer = llm.invoke(answer_prompt.format(**prompt_input))
        if answer is None:
            raise ValueError("LLM returned None for the answer prompt")
        print(f"Answer: {answer}")

        # Return the generated answer along with the SQL result data
        return result_query, result_data, answer

    except Exception as e:
        # Log the exact error encountered during query execution
        print(f"Error during query execution: {e}")
        return f"An error occurred while executing the query: {e}"  # Ensure three values are returned

# FastAPI Setup
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/api/query")
async def handle_query(query: Query):
    result_query, result, answer = execute_query(query.question)
    print(f"Generated SQL result: {result}")
    return {"query": result_query, "result": result, "answer": answer}

# Function to run FastAPI in a separate thread
def run_fastapi():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run FastAPI server in a thread
threading.Thread(target=run_fastapi, daemon=True).start()

# Function to use microphone to capture speech and convert it to text
def listen_for_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening for your voice...")
        audio = recognizer.listen(source, timeout=100, phrase_time_limit=150)  # Adjust timeout and phrase_time_limit
        
    try:
        text = recognizer.recognize_google(audio)
        st.write(f"User: {text}")
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
        return None

st.set_page_config(layout="wide")
st.image("./static/images/logo.png", width=150)

# Chat interface
col1, col2 = st.columns([3, 1])  # Adjust ratios as needed

with col1:
    st.subheader("Products")
    products = [
        {"name": "Insulated Sheet for Window Glass", "price": "¥190.99", "image": "./static/images/window.jpg"},
        {"name": "Gift Pack", "price": "¥290.99", "image": "./static/images/gift.jpg"},
        {"name": "Smart Phone", "price": "¥170.99", "image": "./static/images/phone.jpg"},
        {"name": "Laptop", "price": "¥200.99", "image": "./static/images/laptop.jpg"},
    ]

    for i in range(0, len(products), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(products):
                product = products[i + j]
                with cols[j]:
                    st.image(product["image"], width=150)
                    st.write(product["name"])
                    st.write(product["price"])
                    if st.button("Add to Cart", key=product["name"]):
                        st.success(f"{product['name']} added to cart!")

with col2:
    st.markdown("""<div style="display: flex; align-items: center;">
        <i class="fa fa-headset" style="font-size: 40px; margin-right: 10px; color: green;"></i>
        <h3>Chat with Us</h3>
    </div>""", unsafe_allow_html=True)

    USER = "user"
    ASSISTANT = "ai"
    MESSAGES = "messages"

    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [{"actor": ASSISTANT, "payload": "Hi! How can I help you?"}]

    # Display chat messages
    for msg in st.session_state[MESSAGES]:
        with st.chat_message(msg["actor"]):
            st.markdown(msg["payload"], unsafe_allow_html=True)

    # User input at the bottom
    prompt: str = st.chat_input("Enter your question here...")

    # Add the voice input button
    voice_input = st.button("🎤", key="voice_input", help="Click to speak")
    
    if voice_input:
        spoken_text = listen_for_voice()  # Capture and transcribe voice input
        if spoken_text:
            prompt = spoken_text  # Use the transcribed speech as input
            st.write(f"User (Voice Input): {prompt}")  # Show the captured speech in chat

    if prompt:
        # Detect language of user input
        user_language = detect_language(prompt)  # Detect the language of the user's input
        
        # Only translate if the language is not English
        if user_language != 'en':
            translated_text = translate_text(prompt, user_language)  # Translate to detected language
            st.write(f"User (Translated): {translated_text}")  # Show translated text
        else:
            st.write(f"User: {prompt}")
        
        # Add user message to session state
        st.session_state[MESSAGES].append({"actor": USER, "payload": prompt})
        st.chat_message(USER).write(prompt)

        # Send request to FastAPI for a response
        try:
            response = requests.post('http://localhost:8000/api/query', json={"question": prompt})
            print(f"Response from server: {response.json()}")  # Print the server's response for debugging

            # Process response
            result_query = response.json().get("query")
            result_data = response.json().get("result")
            answer = response.json().get("answer")
            if user_language != "en":
                translated_answer = translate_text(answer, user_language)
                st.session_state[MESSAGES].append({"actor": ASSISTANT, "payload": translated_answer})
                st.chat_message(ASSISTANT).write(translated_answer)
            else:
                st.session_state[MESSAGES].append({"actor": ASSISTANT, "payload": answer})
                st.chat_message(ASSISTANT).write(answer)

        except Exception as e:
            print(f"Error during request: {e}")
            st.session_state[MESSAGES].append({"actor": ASSISTANT, "payload": "Sorry, there was an error processing your request."})
            st.chat_message(ASSISTANT).write("Sorry, there was an error processing your request.")
        
# Add CSS for better layout and position
st.markdown(
    """
    <style>
    .streamlit-expanderHeader {
        display: none;
    }
    .stMain {
        overflow: hidden;
    }
    .stMainBlockContainer {
        padding: 3rem 1rem 10rem; /* Adjust the padding value as needed */
    }

    .stColumn:last-child {
        height: 360px;
        overflow-y: auto; /* Enable scrolling */
    }
    .stChatInput {
        position: fixed;
        bottom: 20px;  /* Distance from the bottom */
        right: 10px;   /* Distance from the right */
        height: 50px;  /* Smaller height */
        font-size: 12px;  /* Smaller font size */
        width: 25%;
        max-width: 400px;
        border: 2px solid green;  /* Green border */
        padding: 5px 40px 5px 10px;
        box-sizing: border-box;
        background-color: transparent;
        overflow-y: auto;
        min-height: 50px; 
        max-height: 150px; 
        resize: vertical;
        z-index: 10;
    }
    
     /* Remove the red border that can appear due to validation */
    .stChatInput:invalid {
        border: 2px solid green
    }

    /* Prevent hover border change */
    .stChatInput:hover {
        border: 2px solid green; /* Prevent hover border change */
        box-sizing: border-box;
    }
    
    /* Mic button inside the chat input */
    .voice_input {
        background-color: transparent;
        border: none;
        color: green;
        font-size: 20px;
        cursor: pointer;
        position: absolute;
        top: 50%;   /* Vertically center it */
        right: 10px; /* Position it at the right edge */
        transform: translateY(-50%); /* Correct vertical alignment */
        z-index: 10;  /* Ensure it's on top of other elements */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Inject JavaScript for automatic scrolling
st.components.v1.html(
    """
    <script>
    const chatContainer = parent.document.querySelector('.stColumn:last-child');
    if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    </script>
    """,
    height=0,
    width=0
)
