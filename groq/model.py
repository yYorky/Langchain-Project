import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

load_dotenv()

# load the GROQ And OpenAI API KEY 
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')

# Add customization options to the sidebar
st.sidebar.title('Customization')
model = st.sidebar.selectbox(
    'Choose a model',
    ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
)
conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

# Initialize conversation memory
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

# Function to save chat history to memory
def save_to_memory(input_text, output_text):
    memory.save_context({'input': input_text}, {'output': output_text})

# Function to retrieve chat history from memory
def retrieve_from_memory():
    chat_history = st.session_state.get('chat_history', [])
    for message in chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

# Check if chat history exists in session state and load it into memory
if 'chat_history' in st.session_state:
    retrieve_from_memory()

# Function for document embedding
def vector_embedding():
    if "vectors" not in st.session_state:

        st.session_state.embeddings = OpenAIEmbeddings()
        #st.session_state.embeddings = OllamaEmbeddings(model="llama3")
        st.session_state.loader=PyPDFDirectoryLoader("./groq/pokemon guide") ## Data Ingestion from pdf folder
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting, use this to split only first 20 docs "st.session_state.docs[:20]""
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector Ollama embeddings


        
# Displaying a GIF
st.image("https://cdn.dribbble.com/users/745569/screenshots/4009638/ava_ai.gif", use_column_width=True)

st.title("Professor Chatgroq with Llama3 for Pokemon Scarlet & Violet (documents)")

# Session state variable
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display conversation history
st.write("Conversation History:")
for message in st.session_state.chat_history:
    st.write("User:", message['human'])
    st.write("Professor ChatGroq:", message['AI'])
    st.write("--------------------")

llm = ChatGroq(groq_api_key=groq_api_key, model_name=model)


# Combine chat history with the current input
chat_history_str = "\n".join(["User: " + message['human'] + "\nAI: " + message['AI'] for message in st.session_state.chat_history])

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Imagine you are a Professor Chatgroq, an expert on pokemon, who is there to help the user.
    Answer the questions based on the provided context only.
    Answer in a natural conversational way.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Chat History:
    {chat_history}
    Questions:{input}
    """
)

prompt1=st.text_input("Ask Your Question")




if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()

    # Invoke the retrieval chain
    response = retrieval_chain.invoke({'input': prompt1, 'chat_history': chat_history_str})

    # Extract the AI response from the response
    ai_response = response['answer']

    # Save the user input and AI response to the chat history
    message = {'human': prompt1, 'AI': ai_response}
    st.session_state.chat_history.append(message)

    print("Response time:", time.process_time() - start)
    st.write(ai_response)

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")