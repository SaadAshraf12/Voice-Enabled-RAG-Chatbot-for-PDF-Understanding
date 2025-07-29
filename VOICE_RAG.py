import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
import requests
import pygame
import sounddevice as sd
import scipy.io.wavfile

from deepgram import DeepgramClient, PrerecordedOptions
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# === ENV CONFIG ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Deepgram client
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

# === AUDIO SETUP ===
DURATION = 5  # seconds
SAMPLE_RATE = 16000  # required for Deepgram

def record_audio(duration=DURATION):
    st.info("🎙️ Recording... Speak now.")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    scipy.io.wavfile.write(wav_path, SAMPLE_RATE, audio)
    return wav_path

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        response = deepgram.listen.rest.v("1").transcribe_file(
            source={"buffer": audio_file, "mimetype": "audio/wav"},
            options=PrerecordedOptions(model="nova-3")
        )
    transcript = response.results.channels[0].alternatives[0].transcript.strip()
    return transcript

# === Deepgram TTS (replacing original) ===
def speak_text(text):
    """Text to speech with better error handling"""
    try:
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        json_data = {
            "text": text
        }

        response = requests.post(url, headers=headers, json=json_data, timeout=30)

        if response.status_code == 200:
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            with open(audio_path, "wb") as f:
                f.write(response.content)

            if not pygame.mixer.get_init():
                pygame.mixer.init()

            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            try:
                os.unlink(audio_path)
            except:
                pass

        else:
            st.error(f"TTS failed with status {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"TTS error: {str(e)}")
        st.info("🔊 Audio playback failed, but here's the response text.")

# === PDF CHUNKING ===
def load_and_chunk_pdfs(file_paths):
    all_chunks = []
    for file_path in file_paths:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = file_path
        all_chunks.extend(chunks)
    return all_chunks

# === EMBEDDING & RETRIEVAL ===
def embed_and_store(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db

# === RAG CHAIN ===
def get_rag_chain(db):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return chain

# === STREAMLIT UI ===
st.set_page_config(page_title="🎙️ Voice RAG Chatbot", layout="centered")
st.title("🎙️ Talk to Your PDFs")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
voice_mode = st.checkbox("🎤 Use Microphone", value=True)

if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_paths.append(tmp_file.name)

    with st.spinner("Processing PDFs..."):
        chunks = load_and_chunk_pdfs(file_paths)
        db = embed_and_store(chunks)
        rag_chain = get_rag_chain(db)

    query = None

    if voice_mode:
        if st.button("🎙️ Record and Ask"):
            audio_path = record_audio()
            query = transcribe_audio(audio_path)
            st.write(f"🗣️ You asked: `{query}`")
    else:
        query = st.text_input("Type your question")

    if query:
        with st.spinner("Thinking..."):
            result = rag_chain({"question": query})
            answer = result['answer']
            sources = result.get("source_documents", [])

            st.markdown("### 🧠 Answer")
            st.write(answer)
            speak_text(answer)

            if sources:
                st.markdown("### 📖 Source Chunks")
                for doc in sources:
                    st.code(f"{doc.page_content}\n\n(Source: {doc.metadata.get('source', '')})")
