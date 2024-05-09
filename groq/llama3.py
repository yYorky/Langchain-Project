import streamlit as st
import os
from langchain_groq import ChatGroq
# from langchain.embeddings import OllamaEmbeddings #deprecated warning
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

## load the GROQ And OpenAI API KEY 
# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv('GROQ_API_KEY')


# Displaying a GIF
st.image("https://cdn.dribbble.com/users/745569/screenshots/4009638/ava_ai.gif", use_column_width=True)


st.title("Professor Chatgroq with Llama3 for Pokemon Scarlet & Violet (documents)")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Imagine you are a Professor Chatgroq,an expert on pokemon, who is there to help the user.
Answer the questions based on the provided context only.
Answer in a natural conversational way.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = OllamaEmbeddings(model="llama3")
        st.session_state.loader=PyPDFDirectoryLoader("./groq/pokemon guide") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #splitting, use this to split only first 20 docs "st.session_state.docs[:20]""
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector Ollama embeddings





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
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")