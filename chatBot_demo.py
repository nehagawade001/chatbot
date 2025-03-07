import os
from urllib.parse import quote
import streamlit as st
import mysql.connector
import requests
import streamlit.components.v1 as components
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
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

import sqlparse
from sqlparse.sql import IdentifierList, Identifier

# Load environment variables
from dotenv import load_dotenv
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

# Initialize SentenceTransformer model for embedding queries
model = SentenceTransformer('all-MiniLM-L6-v2')

# List of Valid Columns
VALID_COLUMNS = {"product_name", "product_description", "product_price", "product_type", "product_category", "image_url", "discount_percent", "product_quantity"}

# Initialize LLM (GoogleGenerativeAI) for query generation
try:
    llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error initializing LLM: {e}")
    llm = None

# Function to fetch table schema dynamically
 
def get_database_schema():
    connection = mysql.connector.connect(user="root", password="root", host="localhost", database="store")
    cursor = connection.cursor()
 
    cursor.execute("SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'store'")
    schema_info = cursor.fetchall()
    
    cursor.close()
    connection.close()
 
    # Organize schema into a dictionary
    schema = {}
    for table, column in schema_info:
        if table not in schema:
            schema[table] = []
        schema[table].append(column)
 
    return schema

def format_schema_for_prompt(schema):
    schema_text = "The database schema contains the following tables and columns:\n\n"
    for table, columns in schema.items():
        schema_text += f"Table `{table}`:\n"
        for column in columns:
            schema_text += f"- `{column}`\n"
        schema_text += "\n"
    schema_text += "Only use tables and columns that exist in the schema above. Do NOT make up column names."
    print(schema_text)
    return schema_text

# Fetch schema once at startup
database_schema = get_database_schema()
schema_text=format_schema_for_prompt(database_schema)


# Prepare the database schema string
# schema_str = "\n".join([f"Table: {table}, Columns: {', '.join(columns)}" for table, columns in database_schema.items()])


# Prepare the few-shot prompt examples for dynamic SQL generation
# Modify the prompt to strictly generate only SQL
# Prepare the few-shot prompt examples for dynamic SQL generation
# - `product_name`
# - `product_description`
# - `product_price`
# - `product_type`
# - `product_category`
# - `image_url`
# - `order_status_name`
# - `order_id`
few_shot_prompt = """Given the following user question, generate a clean and executable SQL query to fetch the relevant product data from the database. Always ensure that the product name (or a similar attribute) is part of the query, even if the question doesn't directly specify it.
From the columns in the database schema generate a valid SQL query based on the user question.
Generate a valid SQL query based on the user question. 

Only use the table names and column names present in following: `schema_text`
Rules:
1. Only use the tables and columns listed above. Do **NOT** make up column names.
2. If a required column is missing, **do not generate the query**.
3. Always check if the condition column exists before using it in WHERE clause.
4. If a column is related to a concept (e.g., "color" for a product), check if it exists first. If not, do not use it
Do NOT generate queries using unknown column names.

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
SQL Query: SELECT product_name, product_description, product_price, image_url FROM products WHERE product_type = 'Smartphone' LIMIT 5;

Question: Do you sell pet food?
SQL Query: SELECT product_name, product_description, product_price, FROM products WHERE product_category LIKE %Pet Food% LIMIT 5;


Question: 300円以下のジャケットが必要です
SQL Query: SELECT product_name, product_description, product_price, FROM products WHERE product_type LIKE %Jacket% LIMIT 5;

Question: Show me jacket for men.
SQL Query: SELECT product_name, product_description, product_price,image_url FROM products WHERE product_type = '%jacket%' LIMIT 5;

Question: Show me red sneakers.
SQL Query: SELECT product_name, product_description, product_price,image_url FROM products WHERE product_type = '%sneakers%' LIMIT 5;

Question: Show me samsung smartphones.
SQL Query: SELECT product_name, product_description, product_price,image_url FROM products WHERE product_type = 'Smartphone' LIMIT 5;

Question: Alchemist book.
SQL Query: SELECT product_name, product_description, product_price,image_url FROM products WHERE product_name LIKE '%Alchemist%';

Question: Image of smartphones.
SQL Query: Select product_name,image_url FROM products WHERE product_type = 'Smartphone';

Question: show me samsung S23.
SQL Query: Select product_name,product_description,image_url FROM products WHERE product_type = 'Smartphone'and product_name LIKE '%Samsung%';

Question: Show me the order information for Logitech Z623 speakers
SQL Query: SELECT products.product_name, orders.order_id, orders.order_status_name FROM products  JOIN orders ON products.product_id = orders.product_id WHERE products.product_name LIKE '%Logitech Z623%' LIMIT 5;

Question: What are the order details for Dell XPS 13 laptop?
SQL Query: SELECT products.product_name, orders.order_id, orders.order_status_name FROM products  JOIN orders  ON products.product_id = orders.product_id WHERE products.product_name LIKE '%Dell XPS 13%' LIMIT 5;

Question: What is the rating of the iPhone 13?
SQL Query: SELECT products.product_name, product_rating_review.product_rating FROM products JOIN product_rating_review ON products.product_id = product_rating_review.product_id WHERE products.product_name LIKE '%iPhone 13%' LIMIT 5;

Question: What is the rating of the Anker PowerCore 10000?
SQL Query: SELECT products.product_name, product_rating_review.product_rating FROM products JOIN product_rating_review ON products.product_id = product_rating_review.product_id WHERE products.product_name LIKE '%Anker PowerCore 10000%' LIMIT 5;

Question: Show me reviews for Samsung Galaxy S21.
SQL Query: SELECT products.product_name, product_rating_review.product_review FROM products JOIN product_rating_review ON products.product_id = product_rating_review.product_id WHERE products.product_name LIKE '%Samsung Galaxy S21%' LIMIT 5;

Question: {question}
SQL Query:  # Only return the SQL query, with no extra explanations or formatting
"""

# Define the PromptTemplate for answering
# answer_prompt = PromptTemplate.from_template(
#     """Given the following user question, SQL query, and query result, provide a human-readable answer:
# - Image: [Click here to view](image_url) (Include only if `image_url` exists)
# Question: {question}
# SQL Query: {query}
# SQL Result: {result}
# Answer: """ `product_name`, `product_category`, `product_description`, `product_type`, `product_price`, `discount_percent`, `image_url`, `product_detail_url`, `product_store_name`,`product_quantity`. 
# )
detected_language = "en"  # Example: Japanese in {detected_language}
answer_prompt = PromptTemplate.from_template(
    f"""Given the following user question, SQL query, and query result, provide a human-readable answer  in the form of a list of bullet points.
        Ensure the query strictly uses only these column from these: {schema_text}
        If the user question refers to attributes that do not exist in the database, return a query that fetches no results.
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

# Function to fetch product data from MySQL
def fetch_product_data():
    query = "SELECT * FROM products"
    connection = mysql.connector.connect(user=user, password=password, host=host, port=port, database=database)
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return result

def clean_sql_query(query):
    # Remove leading and trailing triple backticks (```sql``` and ```)
    query = query.strip()  # First, strip any leading/trailing whitespace
    query = re.sub(r'^```sql\s*', '', query,flags=re.IGNORECASE)  # Remove leading ```sql if present
    query = re.sub(r'```$', '', query)  # Remove trailing ``` if present
    return query

# # Function to Extract Column Names from SQL Query
# def extract_columns_from_query(query):
#     """Extract column names from the SQL query."""
#     match = re.search(r"SELECT (.+?) FROM", query, re.IGNORECASE)
#     if match:
#         columns_part = match.group(1)
#         columns = [col.strip().split(" ")[0] for col in columns_part.split(",")]
#         return set(columns)
#     return set()
 
# # Function to Validate SQL Columns
# def validate_sql_columns(query):
#     """Checks if all columns in the query are valid."""
#     extracted_columns = extract_columns_from_query(query)
#     invalid_columns = extracted_columns - VALID_COLUMNS
#     if invalid_columns:
#         print(f"Invalid columns detected: {invalid_columns}")
#         return False
#     return True
 
# # Function to fetch table schema dynamically
# def get_database_schema():
#     try:
#         connection = mysql.connector.connect(
#             user="root", password="root", host="localhost", port=3306, database="store"
#         )
#         cursor = connection.cursor(dictionary=True)
 
#         query = """
#         SELECT TABLE_NAME, COLUMN_NAME
#         FROM INFORMATION_SCHEMA.COLUMNS
#         WHERE TABLE_SCHEMA = 'store';
#         """
#         cursor.execute(query)
#         schema_data = cursor.fetchall()
 
#         cursor.close()
#         connection.close()
 
#         # Organize schema into a dictionary {table_name: [columns]}
#         schema = {}
#         for row in schema_data:
#             table = row["TABLE_NAME"]
#             column = row["COLUMN_NAME"]
#             if table not in schema:
#                 schema[table] = []
#             schema[table].append(column)
 
#         return schema
 
#     except Exception as e:
#         print(f"Error fetching database schema: {e}")
#         return {}

import re
 
# # Fetch schema once at startup
# database_schema = get_database_schema()

def validate_sql_query(query):
    """
    Ensure that the SQL query only references existing columns in the database.
    """
    global database_schema

    # Extract table and column names using regex
    column_pattern = re.findall(r"[\w]+\.[\w]+", query)  # Matches 'table.column'
    
    for col in column_pattern:
        table, column = col.split('.')
        if table in database_schema:
            if column not in database_schema[table]:
                print(f"Invalid column '{column}' in table '{table}'")
                return False, f"Invalid column '{column}' in table '{table}'."
        else:
            return False, f"Invalid table '{table}'."
 
    return True, None


def execute_query(query):
    try:
        # schema = get_database_schema()

        # Generate the SQL query using the LLM and the user's question
        result_query = chain.run(question=query)  # This gets the SQL query from the LLM
        print(f"Generated SQL Query: {result_query}")  # Log the generated query

        # Clean the SQL query to remove extra text or formatting
        cleaned_query = clean_sql_query(result_query)
        print(f"Cleaned SQL Query: {cleaned_query}")  # Log the cleaned query

        # Ensure only one query is passed to execute
        cleaned_query = cleaned_query.strip()  # Remove any leading/trailing whitespaces

        # Validate if query contains only allowed columns
        # if not validate_sql_columns(cleaned_query):
        #     print("Query is validated")
        #     return "Invalid query generated. Please try again."
        # if not is_valid_query(cleaned_query, schema):
        #     return cleaned_query,[],"Error: The query references non-existent columns. Please refine your question."

        # Validate SQL query against database schema
        is_valid, error_message = validate_sql_query(cleaned_query)
        if not is_valid:
            return f"Error: {error_message}", [], ""

        # Execute the query in the MySQL database
        connection = mysql.connector.connect(user=user, password=password, host=host, port=port, database=database)
        cursor = connection.cursor(dictionary=True)
        cursor.execute(cleaned_query)  # Execute the cleaned query
        
        # Fetch the results of the query
        result = cursor.fetchall()
        print(f"Query result: {result}")  # Log the result

        cursor.close()
        connection.close()

        # if not result:
        #     no_result_message="No matching products were found."
        #     print(no_result_message)
        #     return cleaned_query,[], no_result_message

        # Prepare the result data to be used in the template
        result_data = []
        for row in result:
            product_info = {
                "product_name": row["product_name"],
                "product_description": row.get("product_description", ""),
                "product_price": f"¥{float(row.get('product_price', 0.0)):.2f}" if row.get('product_price') is not None else "N/A",
              #  ": row.get(", "#"),
                "image_url": row.get("image_url", ""),  # Include image URL
                "discount_percent": row.get("discount_percent", 0),
                "product_rating": row.get("product_rating", ""),
                "product_review": row.get("product_review", ""),

            }

            if "order_status_name" in row:
                product_info["order_status"] = row["order_status_name"]
            result_data.append(product_info)

            if not product_info["image_url"]:
                product_info["image_url"]="NO image available"

            result_data.append(product_info)

        # Format the result data into a human-readable string
        # formatted_results = "Product Info:\n"
        # if result_data:
        #     for item in result_data:
        #         formatted_results += f"* <a href=\"{item[']}\">{item['product_name']}</a><br>"
        #         formatted_results += f"    Description: {item['product_description']}<br>"
        #         formatted_results += f"    Price: ${item['product_price']:.2f}<br>"
        #         if item['discount_percent'] is not None:
        #             formatted_results += f"    Discount: {item['discount_percent']}%<br>"
        #         formatted_results += "<br>"  # Add this to separate each product
        # else:
        #     formatted_results = "No matching products found for the given query."

        formatted_results = ""
        # for item in result_data:
        #     formatted_results += f"- **{item['product_name']}**: {item['product_description']}<br>"
        #     formatted_results += f"- **Price**: {item['product_price']}<br>"
            
        #     # Include clickable image link if available
        #     if item["image_url"]:
        #         formatted_results += (
        #             #  f'<a href="{product_page}" target="_blank">'
        #              f'<img src="{item["image_url"]}" width="250"></a><br>' 
        #             )
        #     formatted_results += "<br>"

        # for product in result_data:
        #     image_status = product["image_url"] if product["image_url"] else "No image available"
    
        #     product_details = (
        #     f"**{product['product_name']}**\n"
        #     f"{product.get('product_description', 'No description available.')}\n"
        #     f"**Price:** {product.get('product_price', 'N/A')}\n"
        #     f"**Image:** {image_status}"
        #     )
    
        # formatted_results += f"\n\n{product_details}"  # Append to final response

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
    #return {"query": result_query, "result": result}
    return {"query": result_query, "result": result, "answer": answer}

# Function to run FastAPI in a separate thread
def run_fastapi():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run FastAPI server in a thread
threading.Thread(target=run_fastapi, daemon=True).start()

st.set_page_config(layout="wide")
st.image("./static/images/logo.png", width=150)
#st.title("Cainz Store")

# Add Font Awesome CDN
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True
)
# Streamlit UI for product display and chat
col1, col2 = st.columns([3, 1])  # Adjust ratios as needed

with col1:
    st.subheader("Products")

    products = [
        {"name": "Insulated Sheet for Window Glass", "price": "¥190.99", "image": "./static/images/window.jpg"},
        {"name": "Gift Pack ", "price": "¥290.99", "image": "./static/images/gift.jpg"},
        {"name": "Smart Phone", "price": "¥170.99", "image": "./static/images/phone.jpg"},
        {"name": "Laptop", "price": "¥200.99", "image": "./static/images/laptop.jpg"},
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
    st.markdown("""
        <div style="display: flex; align-items: center;">
           <i class="fa fa-headset" style="font-size: 40px; margin-right: 10px; color: green;"></i>
            <h3>Chat with Us</h3>
        </div>
    """, unsafe_allow_html=True)

    USER = "user"
    ASSISTANT = "ai"
    MESSAGES = "messages"

    if MESSAGES not in st.session_state:
        st.session_state[MESSAGES] = [{"actor": ASSISTANT, "payload": "Hi! How can I help you?"}]

    # Add a unique ID and class to the chat container
    st.markdown('<div id="chat-container" class="chat-container">', unsafe_allow_html=True)

    # Display chat messages
    for msg in st.session_state[MESSAGES]:
        with st.chat_message(msg["actor"]):
            st.markdown(msg["payload"], unsafe_allow_html=True)

    # Close the chat container div
    st.markdown('</div>', unsafe_allow_html=True)

    # User input at the bottom
    prompt: str = st.chat_input("Enter your question here...")

    if prompt:
        # Add user message to session state
        st.session_state[MESSAGES].append({"actor": USER, "payload": prompt})
        st.chat_message(USER).write(prompt)

        # Send request to FastAPI for a response
        try:
            response = requests.post('http://localhost:8000/api/query', json={"question": prompt})
            print(f"respone: {response}")
            if response.ok:
                data = response.json()
                print(f"Received data from FastAPI: {data}") 
                answer = data.get('answer', 'Sorry, I did not understand that.')
                st.session_state[MESSAGES].append({"actor": ASSISTANT, "payload": answer})
    
                # Display the answer
                product_page="https://g.co/kgs/TJt6s1K"
                with st.chat_message(ASSISTANT):
                    st.markdown(answer, unsafe_allow_html=True)
                    # if "image_url" in data["result"][0]:  # To check at least one product has an image
                    #     for product in data["result"]:
                    #         if product["image_url"]:
                    #             image_html = f'<a href="{product_page}" target="_blank">' \
                    #                          f'<img src="{product["image_url"]}" width="250"></a>'\
                    #                         #  f'<p><b>{product["product_name"]}<b></p>'
                    #             st.markdown(image_html, unsafe_allow_html=True)
                            # else:
                            #     st.markdown("<i>No image available</i>",unsafe_allow_html=True)
                    # for product in data.get("result",[]):
                    #     # st.markdown(f"**{product['product_name']}**") 
                    #     # st.markdown(product['product_description'])
                    #     # st.markdown(f"**Price:** {product['product_price']}")

                        # if product.get("image_url"):
                        #     st.image(product["image_url"],width=200)
                        # else:
                        #     st.write("No image available.")
                    # st.markdown(answer, unsafe_allow_html=True)
                    
                # Scroll to the bottom after adding the new message
                st.components.v1.html(
                    """
                    <script>
                     console.log("scrollToBottom function called");
                    function scrollToBottom() {
                        console.log("scrollToBottom function called");
                        const matches = parent.document.querySelectorAll("[data-testid='stChatMessage']");
                        var chatMessages = document.getElementsByClassName('stChatMessage');
                        console.log("matches",matches,matches.length);
                        if (matches.length > 0) {
                            var lastMessage = matches[matches.length - 1];
                            lastMessage.scrollIntoView({ behavior: 'smooth' });
                        }
                    }
                    // Call the function to scroll to the bottom
                    scrollToBottom();
                    </script>
                    """
                )
            else:
                st.session_state[MESSAGES].append({"actor": ASSISTANT, "payload": "Error communicating with the server."})
                st.chat_message(ASSISTANT).write("Error communicating with the server.")
        except Exception as e:
            st.session_state[MESSAGES].append({"actor": ASSISTANT, "payload": f"Error: {str(e)}"})
            st.chat_message(ASSISTANT).write(f"Error: {str(e)}")

# Add CSS to fix the input field at the bottom aligned with the chat messages
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
        # padding: 3rem 1rem 10rem; /* Adjust the padding value as needed */
        position: fixed;
    }

    .stColumn:last-child {
        height: 360px;
        overflow-y: scroll;
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
        padding: 5px 10px;  /* Reduced padding */
        box-sizing: border-box;
        background-color: transparent;
        overflow-y: auto;
        min-height: 50px; 
        max-height: 150px; 
        resize: vertical;
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

    </style>
    """,
    unsafe_allow_html=True
)