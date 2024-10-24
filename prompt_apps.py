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

load_dotenv()

# Database connection parameters
user = "root"
password = "root@123456"
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

# Store embeddings in FAISS
def store_embeddings_in_faiss(products):
    texts = []
    ids = []
    for product in products:
        product_id = str(product['product_id'])
        product_text = f"{product['product_name']} {product['product_description']}  {product['product_category']} {product['product_price']} {product['created_at']}"
        texts.append(product_text)
        ids.append(product_id)

    embeddings_list = model.encode(texts)
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

# Define the PromptTemplate for answering
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
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

def execute_query(question):
    try:
        # Generate SQL query from question
        sql_query = chain.invoke({"question": question})

        # Print the raw generated SQL query for debugging
        st.write("Raw Generated SQL Query:")
        st.code(sql_query)

        # Clean up the SQL query
        cleaned_query = clean_sql_query(sql_query)

        # Print the SQL query for debugging
        st.write("Generated SQL Query:")
        st.code(cleaned_query)

        # Execute the query
        result = db.run(cleaned_query)

        # Prepare the prompt for answering
        prompt_input = {
            "question": question,
            "query": cleaned_query,
            "result": result
        }

        # Generate the answer using the prompt template and llm
        answer = llm.invoke(answer_prompt.format(**prompt_input))

        # Return the cleaned query, result, and answer
        return cleaned_query, result, answer
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None, None

# Message class definition
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
    cleaned_query, result, answer = execute_query(query.question)
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
        st.chat_message(msg.actor).write(msg.payload)

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
                st.chat_message(ASSISTANT).write(answer)
            else:
                st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload="Error communicating with the server."))
                st.chat_message(ASSISTANT).write("Error communicating with the server.")
        except Exception as e:
            st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=f"Error: {str(e)}"))
            st.chat_message(ASSISTANT).write(f"Error: {str(e)}")

    # Call the scroll function after adding messages
    # scroll_to_bottom()

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
