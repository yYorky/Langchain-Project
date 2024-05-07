![](https://python.langchain.com/img/brand/wordmark.png)

# Langchain-Project
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