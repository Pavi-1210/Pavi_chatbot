import streamlit as st
import os
import re
import time
from dotenv import load_dotenv
from langchain_community.llms import Cohere
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import pdfplumber

# Load environment variables
load_dotenv()

# Set environment variables for the APIs
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Vector DB Setup
persist_directory = 'db/pdf_vector_db'
sanitized_video_name = re.sub(r'[^a-z0-9\-]', '-', persist_directory.lower())
persist_directory = sanitized_video_name
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if persist_directory not in existing_indexes:
    pc.create_index(
        name=persist_directory,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

while not pc.describe_index(persist_directory).status["ready"]:
    time.sleep(1)

index = pc.Index(persist_directory)
embedding = HuggingFaceEmbeddings()

# Load the vector store
pdf_vectordb = PineconeVectorStore(index_name=persist_directory, embedding=embedding)

# Define the embedding function
def embed(text):
    global pdf_vectordb
    pdf_vectordb = PineconeVectorStore.from_documents(text, embedding, index_name=persist_directory)
    pdf_vectordb = PineconeVectorStore(index_name=persist_directory, embedding=embedding)
    return "done"

# Function to extract text from PDF using pdfplumber
def extract_text(file_path):
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return "Successfully Uploaded"

# Function to handle question
def handle_question(question, history):
    # Combine all previous questions and responses into the context
    combined_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    combined_input = f"{combined_history}\nQ: {question}"
    
    # Modify the prompt template to include the history
    full_prompt = template.replace("{context}", combined_input).replace("{question}", question)
    
    chain = (
        {"context": pdf_retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    answer = chain.invoke(full_prompt)
    return answer

# Setup the LLM model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)

# Create prompt template
template = """
 
You are a highly intelligent AI assistant. Your primary goal is to provide comprehensive, informative, and accurate responses to user queries, leveraging the provided context. 

**Guidelines:**

1. **Be Clear and Concise:**
   - Use simple, direct, and precise language.
   - Structure your responses in a logical and easy-to-follow manner.

2. **Avoid Speculation:**
   - Base your answers strictly on the provided context.
   - If the context does not provide an answer, politely state that you do not have enough information to respond accurately.

3. **Be Helpful:**
   - If the question is unclear, rephrase it to better understand the user's intent.
   - Offer additional relevant information or context where appropriate to enhance the user's understanding.

4. **Follow ISO Standards:**
   - Ensure all responses adhere to applicable ISO guidelines for clarity, accuracy, and reliability.
   - Provide references to ISO standards where applicable to support your responses.

5. **Engage Polietly:**
   - Greet users when they greet you.
   - Maintain a professional and courteous tone throughout the interaction.

**Format:**

*Context:*

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

pdf_retriever = pdf_vectordb.as_retriever(search_kwargs={"k": 3})

# Streamlit UI
st.title("Document Processing and Question Answering with AI")

# Sidebar for file upload
st.sidebar.header("Upload a PDF File")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Uploading and processing file..."):
        # Save the file to a temporary location
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text
        extracted_text = extract_text(file_path)
        
        # Process and embed the document text
        document = Document(page_content=extracted_text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents([document])
        embed(text_chunks)
        
    st.success("File processed successfully!")
    st.write(extracted_text)

# Input for the question at the top
st.header("Ask a Question")
question = st.text_input("Enter your question", key="question_input")

# Initialize session state to keep track of history
if "history" not in st.session_state:
    st.session_state.history = []

# Button to get the response
if st.button("Get Response"):
    if question:
        with st.spinner("Fetching response..."):
            response = handle_question(question, st.session_state.history)
            st.session_state.history.insert(0, (question, response))  # Insert at the beginning
    else:
        st.error("Please enter a question.")

# Display the entire conversation history
for q, resp in st.session_state.history:
    st.markdown("---")
    st.write(f"**Question:** {q}")
    st.write(f"**Response:** {resp}")


# import streamlit as st
# import os
# import re
# import time
# from dotenv import load_dotenv
# from langchain_community.llms import Cohere
# from langchain_core.runnables import RunnablePassthrough
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.document_loaders import CSVLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# import pdfplumber

# # Load environment variables
# load_dotenv()

# # Set environment variables for the APIs
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")
# os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
# os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

# # Initialize Pinecone
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# # Vector DB Setup
# persist_directory = 'db/pdf_vector_db'
# sanitized_video_name = re.sub(r'[^a-z0-9\-]', '-', persist_directory.lower())
# persist_directory = sanitized_video_name
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

# if persist_directory not in existing_indexes:
#     pc.create_index(
#         name=persist_directory,
#         dimension=768,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )

# while not pc.describe_index(persist_directory).status["ready"]:
#     time.sleep(1)

# index = pc.Index(persist_directory)
# embedding = HuggingFaceEmbeddings()

# # Load the vector store
# pdf_vectordb = PineconeVectorStore(index_name=persist_directory, embedding=embedding)

# # Define the embedding function
# def embed(text):
#     global pdf_vectordb
#     pdf_vectordb = PineconeVectorStore.from_documents(text, embedding, index_name=persist_directory)
#     pdf_vectordb = PineconeVectorStore(index_name=persist_directory, embedding=embedding)
#     return "done"

# # Function to extract text from PDF using pdfplumber
# def extract_text(file_path):
#     text = ''
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + '\n'
#     return text

# # Function to rewrite the query
# def rewrite_query(question):
#     # Basic query rewriting logic: 
#     # This could be a call to an LLM like Cohere or OpenAI for more sophisticated rewriting.
#     if "what is" in question.lower():
#         rewritten_question = question.lower().replace("what is", "please provide information on")
#     elif "who is" in question.lower():
#         rewritten_question = question.lower().replace("who is", "tell me about")
#     else:
#         rewritten_question = question
    
#     return rewritten_question.capitalize()

# # Function to handle question
# def handle_question(question, history):
#     # Rewrite the question before processing it
#     rewritten_question = rewrite_query(question)
    
#     # Combine all previous questions and responses into the context
#     combined_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
#     combined_input = f"{combined_history}\nQ: {rewritten_question}"
    
#     # Modify the prompt template to include the history
#     full_prompt = template.replace("{context}", combined_input).replace("{question}", rewritten_question)
    
#     chain = (
#         {"context": pdf_retriever, "question": RunnablePassthrough()}
#         | prompt
#         | model
#         | StrOutputParser()
#     )
#     answer = chain.invoke(full_prompt)
#     return answer

# # Setup the LLM model
# model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)

# # Create prompt template
# template = """
 
# You are a highly intelligent AI assistant. Your primary goal is to provide comprehensive, informative, and accurate responses to user queries, leveraging the provided context. 

# **Guidelines:**

# 1. **Be Clear and Concise:**
#    - Use simple, direct, and precise language.
#    - Structure your responses in a logical and easy-to-follow manner.

# 2. **Avoid Speculation:**
#    - Base your answers strictly on the provided context.
#    - If the context does not provide an answer, politely state that you do not have enough information to respond accurately.

# 3. **Be Helpful:**
#    - If the question is unclear, rephrase it to better understand the user's intent.
#    - Offer additional relevant information or context where appropriate to enhance the user's understanding.

# 4. **Follow ISO Standards:**
#    - Ensure all responses adhere to applicable ISO guidelines for clarity, accuracy, and reliability.
#    - Provide references to ISO standards where applicable to support your responses.

# 5. **Engage Polietly:**
#    - Greet users when they greet you.
#    - Maintain a professional and courteous tone throughout the interaction.

# **Format:**

# *Context:*

# Context: {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# pdf_retriever = pdf_vectordb.as_retriever(search_kwargs={"k": 3})

# # Streamlit UI
# st.title("Document Processing and Question Answering with AI")

# # Sidebar for file upload
# st.sidebar.header("Upload a PDF File")
# uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# if uploaded_file is not None:
#     with st.spinner("Uploading and processing file..."):
#         # Save the file to a temporary location
#         file_path = os.path.join("data", uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         # Extract text
#         extracted_text = extract_text(file_path)
        
#         # Process and embed the document text
#         document = Document(page_content=extracted_text)
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         text_chunks = text_splitter.split_documents([document])
#         embed(text_chunks)
        
#     st.success("File processed successfully!")
#     st.write(extracted_text)

# # Input for the question at the top
# st.header("Ask a Question")
# question = st.text_input("Enter your question", key="question_input")

# # Initialize session state to keep track of history
# if "history" not in st.session_state:
#     st.session_state.history = []

# # Button to get the response
# if st.button("Get Response"):
#     if question:
#         with st.spinner("Fetching response..."):
#             response = handle_question(question, st.session_state.history)
#             st.session_state.history.insert(0, (question, response))  # Insert at the beginning
#     else:
#         st.error("Please enter a question.")

# # Display the entire conversation history
# for q, resp in st.session_state.history:
#     st.markdown("---")
#     st.write(f"**Question:** {q}")
#     st.write(f"**Response:** {resp}")
