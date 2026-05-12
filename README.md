\# 🎬 AI YouTube Video Intelligence Assistant



An AI-powered YouTube video analysis application built using Streamlit, NVIDIA Llama 3, FAISS, and LangChain.



This project can:



\* Extract transcripts from YouTube videos

\* Convert Hindi transcripts into English

\* Generate AI-powered summaries

\* Answer user questions about the video using RAG (Retrieval-Augmented Generation)

\* Create semantic vector embeddings for intelligent search



\---



\# 🚀 Features



\* 🔗 YouTube video transcript extraction

\* 🌍 Automatic Hindi → English transcript translation

\* 🧠 AI-generated structured summaries

\* 💬 Ask questions about any video

\* 📚 FAISS vector database for semantic retrieval

\* 🔎 Embedding-based context search

\* ⚡ NVIDIA Llama 3 inference API integration

\* 🎨 Clean Streamlit UI



\---



\# 🏗️ Architecture



```text

YouTube URL

&#x20;  ↓

Transcript Extraction

&#x20;  ↓

Hindi/English Detection

&#x20;  ↓

Translation to English

&#x20;  ↓

Text Chunking

&#x20;  ↓

Embeddings Generation

&#x20;  ↓

FAISS Vector Database

&#x20;  ↓

Similarity Search (RAG)

&#x20;  ↓

Llama 3 Response Generation

&#x20;  ↓

Summary + Q\&A

```



\---



\# 🛠️ Tech Stack



\## Frontend



\* Streamlit



\## AI / NLP



\* NVIDIA Llama 3.1 8B

\* LangChain

\* Sentence Transformers



\## Vector Database



\* FAISS



\## Backend / Utilities



\* Python

\* YouTube Transcript API

\* dotenv



\---



\# 📦 Installation



Clone the repository:



```bash

git clone <your-repository-link>

cd <project-folder>

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\# 🔑 Environment Variables



Create a `.env` file and add:



```env

NVIDIA\_API\_KEY=your\_api\_key\_here

```



\---



\# ▶️ Run the Project



```bash

streamlit run app.py

```



\---



\# 💡 How It Works



1\. User enters a YouTube video URL

2\. Transcript is extracted using YouTube Transcript API

3\. Hindi transcripts are translated into English

4\. Transcript is split into chunks

5\. Embeddings are generated using MiniLM

6\. FAISS stores embeddings for semantic search

7\. Relevant chunks are retrieved using similarity search

8\. NVIDIA Llama 3 generates summaries and answers



\---



\# 📸 Capabilities



\* Works with both Hindi and English videos

\* Generates English summaries only

\* Provides AI-powered question answering

\* Uses RAG pipeline for contextual responses



\---



\# 🔮 Future Improvements



\* Multi-language support

\* Chat history memory

\* PDF export

\* Timestamp-based navigation

\* Video thumbnail preview

\* Voice-based interaction



\---



\# 🔗 Useful Links



\## GitHub Repository



```text

https://github.com/your-username/your-repository-name

```



\## NVIDIA API



```text

https://build.nvidia.com/

```



\## Streamlit Documentation



```text

https://docs.streamlit.io/

```



\## LangChain Documentation



```text

https://python.langchain.com/

```



\## FAISS Documentation



```text

https://faiss.ai/

```



\---



\# 👩‍💻 Author



Ruchira Devane



