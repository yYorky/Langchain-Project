import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Import HTML templates for chat messages
from htmlTemplates import css, bot_template, user_template

# Load environment variables from .env file
load_dotenv()

# Load API keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Function for document embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings_initialized = False
        st.sidebar.write("Initializing embeddings...")

        if st.session_state.embedding_choice == 'OpenAI':
            st.session_state.embeddings = OpenAIEmbeddings()
        elif st.session_state.embedding_choice == 'Ollama':
            st.session_state.embeddings = OllamaEmbeddings(model="llama3")
        elif st.session_state.embedding_choice == 'GoogleGenerativeAI':
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        st.sidebar.write("Embeddings initialized.")

        st.sidebar.write("Loading documents from PDF directory...")
        st.session_state.loader = PyPDFDirectoryLoader("./groq/pokemon guide")  # Data Ingestion from PDF folder
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.sidebar.write(f"{len(st.session_state.docs)} documents loaded.")

        st.sidebar.write("Splitting documents into chunks...")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.sidebar.write(f"{len(st.session_state.final_documents)} chunks created.")

        st.sidebar.write("Creating vector embeddings...")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.sidebar.write("Vector embeddings created.")
        st.session_state.embeddings_initialized = True

# Function to get the conversational retrieval chain
def get_conversation_chain(vectorstore, model_name):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, max_length=st.session_state.conversational_memory_length, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True  # Ensure source documents are returned
    )
    return conversation_chain

# Add customization options to the sidebar
st.sidebar.title('Customization')
model = st.sidebar.selectbox(
    'Choose a model',
    ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'],
    key='model_choice'  # Unique key for the selectbox
)
st.session_state.embedding_choice = st.sidebar.selectbox(
    'Choose embedding type',
    ['GoogleGenerativeAI', 'OpenAI', 'Ollama'],
    key='embedding_choice_main'  # Unique key for the selectbox in the main sidebar
)
st.session_state.conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

if st.sidebar.button("Documents Embedding"):
    vector_embedding()
    st.sidebar.write("Vector Store DB Is Ready")
    st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectors, model)  # Initialize the conversation chain

# Displaying a GIF
st.image("https://cdn.dribbble.com/users/745569/screenshots/4009638/ava_ai.gif", use_column_width=True)

st.title("Professor Chatgroq with Selected Model for Pokemon Scarlet & Violet (documents)")

# Session state variable
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def handle_userinput():
    user_question = st.session_state.user_question
    # st.write("User Question: ", user_question)  # Debug statement
    if not st.session_state.get("embeddings_initialized", False):
        st.write("Please initialize the embeddings first.")
        return
    
    if "conversation_chain" not in st.session_state:
        st.write("Please initialize the conversation chain first.")
        return

    start = time.process_time()
    
    # Invoke the retrieval chain
    response = st.session_state.conversation_chain({'question': user_question, 'chat_history': st.session_state.chat_history})
    # st.write("Response: ", response)  # Debug statement to check response structure
    ai_response = response['answer']

    # Save the user input and AI response to the chat history
    st.session_state.chat_history.append({'human': user_question, 'AI': ai_response})

    # st.write("Response time:", time.process_time() - start)

    # Clear the input box after processing the question
    st.session_state.user_question = ""

    display_chat_history()

    # Store response for document similarity search
    st.session_state.response = response

def display_chat_history():
    st.write(css, unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", message['human']), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message['AI']), unsafe_allow_html=True)

def main():
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    # # Display the chat history only in the designated section
    # st.write('<div style="height: 300px; overflow-y: auto;">', unsafe_allow_html=True)
    # display_chat_history()
    # st.write('</div>', unsafe_allow_html=True)

    st.text_input("Ask Professor Chatgroq any question about Pokemon Scarlet & Violet:", key="user_question", on_change=handle_userinput)

    # With a streamlit expander
    if 'response' in st.session_state:
        with st.expander("Document Similarity Search"):
            # st.write("Document Similarity Search Triggered")  # Debug statement
            # st.write("Full response: ", st.session_state.response)  # Debug statement to print the entire response
            if "source_documents" in st.session_state.response:
                # st.write(f"Number of documents found: {len(st.session_state.response['source_documents'])}")  # Debug statement
                for i, doc in enumerate(st.session_state.response["source_documents"]):
                    # st.write(f"Document {i+1}:")  # Debug statement
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No source documents found in the response.")

if __name__ == '__main__':
    main()
