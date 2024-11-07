import os
from urllib.parse import quote
import streamlit as st
import requests
import streamlit.components.v1 as components
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
import mysql.connector
import validators
import re

load_dotenv()

# Database connection parameters
user = "root"
password = "root"
host = "localhost"
port = 3306
database = "store"

# Create SQL
encoded_password = quote(password)
db_uri = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{database}"

# Initialize SQLDatabase
db = SQLDatabase.from_uri(db_uri)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Fetch product data from SQL
def fetch_product_data():
    query = "SELECT * FROM products"
    connection = mysql.connector.connect(user=user, password=password, host=host, port=port, database=database)
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result

# Convert product data to embeddings using SentenceTransformers
def get_embedding(text):
    return model.encode(text)    

# Store embeddings in FAISS
def store_embeddings_in_faiss(products):
    texts = []
    ids = []
    for product in products:
        product_id = str(product['product_id'])
        product_text = f"{product['product_name']} {product['product_description']}  {product['product_category']} {product['product_price']} {product['created_at']}"
        texts.append(product_text)
        ids.append(product_id)

    embeddings_list = get_embedding(texts)
    embeddings_array = np.array(embeddings_list).astype('float32')

    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)

    return index, ids

# Initialize LLM
try:
    llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    llm = None

# Create SQL query chain
chain = create_sql_query_chain(llm, db)

# Define the PromptTemplate for getting the answer, product detail URL, and image URLs with few-shot examples
answer_and_images_prompt = PromptTemplate.from_template(
    """Given the following user question and the matching product data, answer the user question and provide the product detail URL and image URLs of the matching products.

Examples:
Question: What is the price of the Logitech Z623 speakers?
Matching Products: Logitech Z623 2.1 speaker system with THX certification costs $149.99.
Answer:
Product Info:
  The Logitech Z623 2.1 speaker system with THX certification costs $149.99.
* Logitech Z623<br>
  Product URL: <a href="https://example.com/logitech-z623">https://example.com/logitech-z623</a><br>
  Image: <img src="https://d30uxjjrk95rd.cloudfront.net/img/goods/L/4902430473637.jpg" width="50">

Question: What is Corsair K95?
Matching Products: The Corsair K95 RGB is a mechanical gaming keyboard with customizable RGB lighting.
Answer:
Product Info:
  The Corsair K95 RGB is a mechanical gaming keyboard with customizable RGB lighting.
* Corsair K95 RGB<br>
  Product URL: <a href="https://example.com/corsair-k95">https://example.com/corsair-k95</a><br>
  Image: not available

Question: Show me the Logitech products.
Matching Products: Logitech G502, Logitech G915, Logitech G Pro X Superlight 
Answer:
Product Info:
  There are 3 Logitech products present.
* Logitech G502<br>
  Product URL: <a href="https://example.com/logitech-g502">https://example.com/logitech-g502</a><br>
  Image: <img src="https://d30uxjjrk95rd.cloudfront.net/img/goods/L/4902430473637.jpg" width="50">
* Logitech G915<br>
  Product URL: <a href="https://example.com/logitech-g915">https://example.com/logitech-g915</a><br>
  Image: <img src="https://d30uxjjrk95rd.cloudfront.net/img/goods/L/4902430473637.jpg" width="50">
* Logitech G Pro X Superlight<br>
  Product URL: <a href="https://example.com/logitech-g-pro-x-superlight">https://example.com/logitech-g-pro-x-superlight</a><br>
  Image: <img src="https://d30uxjjrk95rd.cloudfront.net/img/goods/L/4902430473637.jpg" width="50">

Question: {question}
Matching Products: {result}
Answer: """
)

def scroll_to_bottom():
    components.html(
        """
        <script>
            const chatContainer = document.querySelector('.st-emotion-cache-keje6w');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            } else {
                alert("Chat container not found!");
            }
        </script>
        """,
        height=0,
    )

def clean_sql_query(sql_query):
    if sql_query:
        cleaned_query = sql_query.split("SQLQuery:")[-1].strip()
        return cleaned_query.replace("```sql", "").replace("```", "").strip()
    return sql_query

def execute_query(query, index, ids):
    try:
        query_embedding = get_embedding([query]).astype('float32')
        D, I = index.search(np.array(query_embedding), k=10)
        result_ids = [ids[i] for i in I[0]]
        print(result_ids)
        result_query = f"SELECT product_name, product_description, product_category, product_type, product_price, image_url,product_detail_url FROM products WHERE product_id IN ({','.join(map(str, result_ids))})"
        result = db.run(result_query)

        # Prepare the prompt for answering
        prompt_input = {
            "question": query,
            "result": result
        }

        # Generate the answer using the prompt template and llm
        answer_and_images = llm.invoke(answer_and_images_prompt.format(**prompt_input))
        if answer_and_images is None:
            raise ValueError("LLM returned None for the answer and images prompt")
        print(answer_and_images)

        # Return the cleaned query, result, and answer
        return query, result, answer_and_images.strip()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None, None

class Message:
    def __init__(self, actor, payload):
        self.actor = actor
        self.payload = payload

# FastAPI setup
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/api/query")
async def handle_query(query: Query):
    products = fetch_product_data()
    index, ids = store_embeddings_in_faiss(products)
    cleaned_query, result, answer = execute_query(query.question, index, ids)

    return {"query": cleaned_query, "result": result, "answer": answer}

# Function to run FastAPI in a separate thread
def run_fastapi():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run FastAPI server in a thread
threading.Thread(target=run_fastapi, daemon=True).start()

# Streamlit Interface
st.set_page_config(layout="wide")
st.title("My eCommerce Store")

# Fetch product data and store embeddings if not already done
if 'index' not in st.session_state:
    products = fetch_product_data()
    st.session_state.index, st.session_state.ids = store_embeddings_in_faiss(products)

# Create two columns: one for products and one for chat
col1, col2 = st.columns([1, 1])  # Adjust ratios as needed

with col1:
    st.subheader("Products")

    products = [
        {"name": "Smart Watch", "price": "$19.99", "image": "./static/images/smart_watch.jpg"},
        {"name": "Samsung Smart Phone ", "price": "$29.99", "image": "./static/images/samsung_smart_phone.jpg"},
        {"name": "Smart Phone", "price": "$17.99", "image": "./static/images/smart_phone.jpg"},
    ]

    # Create two columns for displaying products side by side
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
    USER = "user"
    ASSISTANT = "ai"
    MESSAGES = "messages"

    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Hi! How can I help you?")]

    # Display chat messages
    for msg in st.session_state[MESSAGES]:
        with st.chat_message(msg.actor):
            st.markdown(msg.payload, unsafe_allow_html=True)

    # User input at the bottom
    prompt: str = st.chat_input("Enter your question here...")

    if prompt:
        # Add user message to session state
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
        st.chat_message(USER).write(prompt)

        # Send request to FastAPI for a response
        try:
            response = requests.post('http://localhost:8000/api/query', json={"question": prompt})
            if response.ok:
                data = response.json()
                answer = data.get('answer', 'Sorry, I did not understand that.')
                st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=answer))
    
                # Display the answer
                with st.chat_message(ASSISTANT):
                    st.markdown(answer, unsafe_allow_html=True)
            else:
                st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload="Error communicating with the server."))
                st.chat_message(ASSISTANT).write("Error communicating with the server.")
        except Exception as e:
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=f"Error: {str(e)}"))
            st.chat_message(ASSISTANT).write(f"Error: {str(e)}")

# Add CSS to fix the input field at the bottom aligned with the chat messages
st.markdown(
    """
    <style>
    .streamlit-expanderHeader {
        display: none;
    }
    .stChatInput {
        position: fixed;
        bottom: 20px; /* Distance from the bottom */
        left: 72%; /* Center horizontally */
        transform: translateX(-50%); /* Shift left by half its width */
        padding: 10px;
        background-color: white; /* Adjust background color if needed */
        width: 80%; /* Set a width relative to the screen */
        max-width: 600px; /* Maximum width for larger screens */
    }
    </style>
    """,
    unsafe_allow_html=True
)