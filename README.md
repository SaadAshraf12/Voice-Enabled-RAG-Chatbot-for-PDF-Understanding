## 🎙️ **Voice-Enabled RAG Chatbot for PDF Understanding**

### 🧩 **Project Overview**

This project presents an intelligent, voice-enabled chatbot that uses Retrieval-Augmented Generation (RAG) to answer user queries based on uploaded PDF documents. It leverages advanced Natural Language Processing (NLP), speech recognition, and text-to-speech technologies to enable a hands-free, interactive user experience. Built with Streamlit for an intuitive UI, the chatbot supports both voice and text input, and provides answers using conversational AI with references to the original document context.

---

### 🧠 **Key Features**

* **Multimodal Input**: Users can ask questions via voice or text.
* **PDF Parsing & Chunking**: Uploaded PDFs are parsed and split into semantically meaningful text chunks.
* **Context-Aware QA**: Answers are generated using retrieval-based LLM chaining to maintain accuracy and contextual relevance.
* **Voice Interaction**: Speech-to-text (Deepgram ASR) and text-to-speech (Deepgram TTS) integration for hands-free interaction.
* **Source Referencing**: Each response includes highlighted source excerpts from the documents.

---

### 🛠️ **Tech Stack**

| Component            | Technology/Library                                       |
| -------------------- | -------------------------------------------------------- |
| Frontend UI          | [Streamlit](https://streamlit.io)                        |
| Speech-to-Text (ASR) | [Deepgram Nova-3 API](https://developers.deepgram.com)   |
| Text-to-Speech (TTS) | [Deepgram Aura API](https://developers.deepgram.com)     |
| Audio Recording      | `sounddevice`, `scipy.io.wavfile`, `pygame`              |
| Document Loader      | `langchain_community.document_loaders.PyMuPDFLoader`     |
| Text Chunking        | `langchain.text_splitter.RecursiveCharacterTextSplitter` |
| Embeddings           | `OpenAIEmbeddings` (via LangChain)                       |
| Vector Store         | [FAISS](https://github.com/facebookresearch/faiss)       |
| Language Model       | `ChatOpenAI (gpt-3.5-turbo)` via LangChain               |
| RAG Chain            | `ConversationalRetrievalChain` from LangChain            |
| Memory               | `ConversationBufferMemory` from LangChain                |

---

### 📌 **Use Cases**

* **Academic Research**: Quickly extract answers from research papers, textbooks, or technical PDFs.
* **Corporate Reports**: Navigate and query large financial or audit reports using natural language.
* **Legal Document Analysis**: Ask legal questions to receive context-based responses from contracts, policies, or case files.
* **Hands-Free Reading Assistance**: Supports users with visual impairments or multitasking needs through voice-based document querying.
* **Customer Support Training**: Upload product manuals or support documents and simulate FAQ interaction using natural voice.

---

### 🔐 **Environment & Security**

* Environment variables (API keys) are securely managed using `dotenv`.
* Temporary files (PDFs, audio recordings) are handled via `tempfile` and automatically cleaned up after use.
* Real-time interaction without storing user queries or documents post-session.

---

### 🚀 **How It Works (Pipeline Overview)**

1. **PDF Upload** → User uploads one or more PDF files.
2. **Text Extraction & Chunking** → PDFs are split into overlapping text chunks with metadata.
3. **Embedding & Storage** → Chunks are embedded and indexed using FAISS.
4. **Query Input** → User speaks or types a question.
5. **Voice to Text** → Deepgram ASR transcribes audio (if voice mode enabled).
6. **RAG Retrieval** → Relevant document chunks are retrieved.
7. **LLM Response** → OpenAI’s GPT-3.5 generates a natural language response.
8. **Text to Speech** → Answer is converted to audio using Deepgram Aura TTS.
9. **Source Display** → The documents supporting the answer are shown with metadata.

---

### 🧪 **Future Enhancements**

* Add support for multilingual speech transcription and translation.
* Enable real-time streaming TTS instead of file-based playback.
* Fine-tune document chunking using semantic segmentation for better QA accuracy.
* Integrate user authentication and session-based memory persistence.
* Option to download QA logs and transcripts for audit or study.
