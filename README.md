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

