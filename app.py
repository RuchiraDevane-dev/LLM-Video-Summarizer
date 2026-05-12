import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

st.set_page_config(
    page_title="AI YouTube Video Summarizer",
    layout="wide"
)

st.title("🎬 AI Video Intelligence Assistant")


def extract_video_id(url):
    pattern = r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()

        transcript = api.fetch(
            video_id,
            languages=["en", "en-US", "en-GB", "hi"]
        )

        transcript_text = ""

        for snippet in transcript:
            start_time = int(snippet.start)
            minutes = start_time // 60
            seconds = start_time % 60
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            transcript_text += f"{timestamp} {snippet.text}\n"

        return transcript_text

    except Exception:
        return None


def translate_to_english(transcript_text):
    prompt = f"""
Translate the following transcript completely into natural English.

Rules:
- Keep timestamps unchanged.
- Translate everything into clear English.
- Do not summarize.
- Do not skip important content.
- Output only English text.

Transcript:
{transcript_text[:12000]}
"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a professional translator. Translate all content into English only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=2500
    )

    return response.choices[0].message.content


def create_vector_store(transcript_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(transcript_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store


def generate_summary(transcript_text):
    prompt = f"""
You are an AI YouTube video summarizer.

IMPORTANT:
- Always write the complete answer only in English.
- Do not use Hindi or Hinglish.
- The transcript is already translated into English.
- Give a clean and structured summary.

Summarize the following transcript accurately.

Include:
1. Short Summary
2. Key Points
3. Important Terms
4. Difficulty Level
5. Who Should Watch This Video
6. Important Timestamp Highlights

Transcript:
{transcript_text[:12000]}
"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You summarize video transcripts accurately. Always respond only in English."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content


def ask_question(vector_store, question, transcript_text):
    docs = vector_store.similarity_search(question, k=5)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI video intelligence assistant.

IMPORTANT:
- Always answer only in English.
- Do not use Hindi or Hinglish.
- The transcript context is already in English.

Your job:
1. First use the transcript context to answer the question.
2. If the question is directly related to the video, answer from the transcript.
3. If the question is general or simple, answer normally and connect it with the video when possible.
4. Do not say "not enough information" unless the question is specifically asking for something that is not in the video.

Transcript Context:
{context}

Full Transcript Preview:
{transcript_text[:6000]}

Question:
{question}

Give a clear and helpful answer only in English.
"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Always answer only in English."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=700
    )

    return response.choices[0].message.content


youtube_url = st.text_input("Enter YouTube Video URL")


if st.button("Generate Summary"):

    if youtube_url:

        video_id = extract_video_id(youtube_url)

        if video_id:

            with st.spinner("Extracting transcript..."):
                original_transcript = get_transcript(video_id)

            if not original_transcript:
                st.warning(
                    "Transcript could not be fetched for this video. "
                    "Please try another video with English or Hindi captions enabled."
                )
                st.stop()

            try:
                with st.spinner("Converting transcript to English..."):
                    english_transcript = translate_to_english(original_transcript)

                with st.spinner("Creating vector database..."):
                    vector_store = create_vector_store(english_transcript)

                with st.spinner("Generating summary using NVIDIA Llama 3..."):
                    summary = generate_summary(english_transcript)

                st.session_state.transcript_text = english_transcript
                st.session_state.vector_store = vector_store
                st.session_state.summary = summary
                st.session_state.video_id = video_id

            except Exception as e:
                st.error("Something went wrong while processing the transcript.")
                st.code(str(e))

        else:
            st.error("Invalid YouTube URL")

    else:
        st.warning("Please enter a YouTube URL")


if "summary" in st.session_state:

    st.success("Transcript Processed Successfully!")

    st.subheader("Video ID")
    st.write(st.session_state.video_id)

    st.subheader("Transcript Length")
    st.write(f"{len(st.session_state.transcript_text.split())} words")

    st.subheader("AI Summary")
    st.write(st.session_state.summary)

    with st.expander("View English Transcript"):
        st.write(st.session_state.transcript_text[:7000])


st.divider()

st.subheader("💬 Ask Questions About the Video")

user_question = st.text_input("Enter your question")

if st.button("Get Answer"):

    if "vector_store" not in st.session_state:
        st.warning("First generate the summary.")

    elif not user_question:
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Generating answer..."):
                answer = ask_question(
                    st.session_state.vector_store,
                    user_question,
                    st.session_state.transcript_text
                )

            st.subheader("Answer")
            st.write(answer)

        except Exception as e:
            st.error("Could not generate answer.")
            st.code(str(e))