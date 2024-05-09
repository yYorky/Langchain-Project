![](https://cdn.hashnode.com/res/hashnode/image/upload/v1709793662749/2d1f2dd1-5c39-4c71-93a6-e14e68664a21.png)

# Langchain-Project (with Ollama)
Langchain Project following Krish Naik's Updated Langchain Tutorial
- link: https://www.youtube.com/playlist?list=PLZoTAELRMXVOQPRG7VAuHL--y97opD5GQ

## Log
### Part 2 of Updated Langchain playlist
1. Create localama.py using ollama and gemma model
2. To run go to venv and type `streamlit run chatbot/localama.py`

### Part 3 of Updated Langchain playlist
1. Create api `app.py` and `client.py` to route different task to different LLM
2. To run go to type `python api/app.py` to activate the api app
3. Next, open another terminal and navigate to venv and type `streamlit run api/client.py` to run the frontend app

### Part 4 of Updated Langchain playlist
1. Ingest data from `.txt`, website, `.pdf`
2. Use Recursive text splitter to split corpus into chunks
3. Vector Embedding using open source model (Ollama = llama3)
4. Store in vector store FAISS
5. perform query using similarity search

### Part 5 of Updated Langchain playlist
1. Combine Chain and Retriever into a retriever_chain to obtain response for LLM model based on context
2. Context are the text information embeddeded and stored in the vector store
3. Some prompts are listed as part of the input to ensure better quality answer suited to the case.

### Part 6 of Updated Langchain playlist
1. Using agents and tools for multi-data source RAG
2. Used Groqchat as it supports tools (https://python.langchain.com/docs/integrations/chat/)
3. Used Groqcloud to obtain API keys
4. Invoking the agents to use the relevant tools requires explicit command on the

### Part 7 of Updated Langchain playlist
1. Groq inferencing engine using open source LLM
2. Stored 3 websites for context, gave better performance with details compared to chatGPT and Copilot

### Part 8 of Updated Langchain playlist
1. Try out Huggingface embedder in notebook

### Part 9 of Updated Langchain playlist
1. Q&A using Llama3 model and RAG on documents
2. Added gif for processing effect
3. change to dark mode.


### Reference documents
- https://docs.google.com/document/d/1Dh13wSb-3SJkKh-I_ihw-K0eMcjrKyLJXQ_ZE1_Pcug/edit
- https://docs.google.com/document/d/1YPP8AQUN6Wbr7xhARJb5l2QzPkfmMGQDBMo9tjqR2wU/edit
- https://docs.google.com/document/d/1xL1NNZnKRabyl93BewLzcZkcvmDJj2612K0Ih10hqXQ/edit#heading=h.82hsajnz9z9b


## Detailed explaination of `groq/llama3.py` script 

The script sets up a Streamlit app where users can ask questions related to Pokémon Scarlet & Violet documents. It prepares document embeddings and retrieves relevant responses using the ChatGroq model and document retrieval techniques.

<p align="center">
  <img src="https://github.com/yYorky/Langchain-Project/blob/main/static/LangChain Project Flowchart.JPG" alt="image"/>
</p>

- **Imports:** The script imports necessary libraries and modules including Streamlit for building the web app, os for handling environment variables, various components from the "langchain" package for language processing, and dotenv for loading environment variables from a .env file.

- **Loading Environment Variables:** It loads environment variables, particularly the GROQ API key, from a .env file using the dotenv library.


- **Initializing ChatGroq:** Creates an instance of the ChatGroq class with the GROQ API key and model name "Llama3-8b-8192".

- **Setting Up Chat Prompt Template:** Defines a template for generating prompts for interacting with the ChatGroq model.

- **Function Definition (vector_embedding):** Defines a function named vector_embedding which sets up document embeddings using Ollama embeddings and FAISS for vector storage.

- **User Input:** Accepts user input via a text input field (st.text_input) for asking questions.

- **Button for Document Embedding:** Displays a button labeled "Documents Embedding". Upon clicking this button, it triggers the vector_embedding function to prepare the document embeddings.

- **Document Retrieval:** If the user inputs a question and submits it, the script initiates document retrieval by creating a retrieval chain using the prepared document embeddings and the ChatGroq model. It then retrieves and displays the response.

- **Document Similarity Search:** It expands an expander section in the Streamlit app to display document similarity search results. This section iterates over the retrieved documents and displays their content along with a separator.