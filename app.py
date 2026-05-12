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
            languages=["en", "en-GB", "hi"]
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


def create_vector_store(transcript_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(transcript_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    return vector_store


def generate_summary(transcript_text):
    prompt = f"""
You are an AI YouTube video summarizer.

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
                "content": "You summarize transcripts accurately."
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


def ask_question(vector_store, question):
    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an AI assistant.

Answer ONLY using the provided transcript context.

If answer is not present, say:
"The transcript does not contain enough information."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {
                "role": "system",
                "content": "Answer accurately from retrieved transcript chunks."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=500
    )

    return response.choices[0].message.content


youtube_url = st.text_input("Enter YouTube Video URL")


if st.button("Generate Summary"):

    if youtube_url:

        video_id = extract_video_id(youtube_url)

        if video_id:

            with st.spinner("Extracting transcript..."):
                transcript_text = get_transcript(video_id)

            if not transcript_text:
                st.warning(
                    "Transcript could not be fetched for this video on the deployed server. "
                    "This usually happens because YouTube blocks cloud server requests. "
                    "Please try another video with captions enabled, or run the project locally."
                )
                st.stop()

            try:
                with st.spinner("Creating vector database..."):
                    vector_store = create_vector_store(transcript_text)

                with st.spinner("Generating summary using NVIDIA Llama 3..."):
                    summary = generate_summary(transcript_text)

                st.session_state.transcript_text = transcript_text
                st.session_state.vector_store = vector_store
                st.session_state.summary = summary
                st.session_state.video_id = video_id

            except Exception:
                st.error(
                    "Something went wrong while processing the transcript. "
                    "Please check your NVIDIA API key and dependencies."
                )

        else:
            st.error("Invalid YouTube URL")

    else:
        st.warning("Please enter a YouTube URL")


if "summary" in st.session_state:

    st.success("Transcript Processed Successfully!")

    st.subheader("Video ID")
    st.write(st.session_state.video_id)

    st.subheader("Transcript Length")
    st.write(
        f"{len(st.session_state.transcript_text.split())} words"
    )

    st.subheader("AI Summary")
    st.write(st.session_state.summary)

    with st.expander("View Transcript"):
        st.write(
            st.session_state.transcript_text[:7000]
        )


st.divider()

st.subheader("💬 Ask Questions About the Video")

user_question = st.text_input(
    "Enter your question"
)

if st.button("Get Answer"):

    if "vector_store" not in st.session_state:

        st.warning(
            "First generate the summary."
        )

    elif not user_question:

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Retrieving relevant transcript chunks..."
            ):

                answer = ask_question(
                    st.session_state.vector_store,
                    user_question
                )

            st.subheader("Answer")
            st.write(answer)

        except Exception:

            st.error(
                "Could not generate answer. Please try again."
            )