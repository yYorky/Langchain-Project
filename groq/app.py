import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3")
    websites = [
        "https://www.nintendolife.com/guides/pokemon-scarlet-and-violet-beginners-guide-best-early-pokemon-provinces-tms",
        "https://www.nintendolife.com/guides/pokemon-scarlet-and-violet-best-starter-pokemon-all-starter-evolutions",
        "https://www.nintendolife.com/guides/pokemon-scarlet-and-violet-cortondo-gym-bug-badge-how-to-beat-leader-katy"
        # Add more URLs here
    ]
    st.session_state.final_documents = []

    for url in websites:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # Split documents and process if necessary
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(docs[:50])
        st.session_state.final_documents.extend(final_docs)

    # Once all documents are loaded and processed, generate vectors
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Professor ChatGroq for Pokemon Scarlet & Violet")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
"""
Imagine you are a Pokemon Professor who is there to help the user.
Answer the questions based on the provided context only.
Answer in a natural conversational way.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input your question here")

if prompt:
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")