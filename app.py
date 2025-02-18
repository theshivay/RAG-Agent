import streamlit as st # For Ui components
from langchain_community.document_loaders import PDFPlumberLoader # For loading the whole pdf document
from langchain_text_splitters import RecursiveCharacterTextSplitter # For splitting the document into chunks
from langchain_core.vectorstores import InMemoryVectorStore # For storing the document vectors (locally)
from langchain_ollama import OllamaEmbeddings # For generating embeddings of the document chunks
from langchain_core.prompts import ChatPromptTemplate # For generating the conversational prompt
from langchain_ollama.llms import OllamaLLM # For generating the response to the user query


# CSS Styking for the Chat Interface
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)


# Prompt template that model will use to generate the response
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""


PDF_STORAGE_PATH = 'document_store/' # Store the path of the uploaded pdf file
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b") # Store the embedding model that use to generate the embeddings of the document chunks
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL) # Store the document vectors in memory(locally)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b") # Store the language model that use to generate the response to the user query


# this helper function saves the uploaded file to the local storage(document_store) and returns the file path
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# this helper function loads the pdf document and returns the raw documents
def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load() 

# this helper function splits the raw documents into chunks and returns the document chunks
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

# this helper function indexes the document chunks into the document vector store
def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

# this helper function finds the related documents based on the user query using similarity search
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)


# this helper function generates the answer to the user query using the language model
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents]) # Combine the context documents
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) # Create the conversation prompt
    response_chain = conversation_prompt | LANGUAGE_MODEL # Create the response chain
    return response_chain.invoke({"user_query": user_query, "document_context": context_text}) # Generate the response using the response chain


# UI Configuration
st.title("📘 DocuMind AI")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False # Only allow one file to be uploaded

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)