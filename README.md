# 🧠 Notesbot Chatbot

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://notesbot-chatbot-hbvpkbtztypne7xyxgcxc8.streamlit.app)

An intelligent, user-friendly chatbot built with **Streamlit** that allows users to upload academic documents (PDFs, DOCX, PPTX, CSV, TXT) and ask questions directly based on their content. Ideal for students and educators looking to quickly understand notes, lectures, and other academic materials.

---

## 🚀 Key Features

- 🔄 **Multi-File Upload**: Upload multiple file types in one session (PDF, DOCX, PPTX, CSV, TXT).
- 💬 **Contextual Q&A**: Ask questions based on the content of uploaded or saved documents.
- 📌 **Source Reference**: Cites the filename and page number at the end of each answer.
- 🗂 **Permanent Vectorstore**: Stores document vectors so you don't need to reprocess every time.
- 🧹 **Clear Chat History**: Reset conversations for a fresh session.
- 💾 **Export Chat**: Download entire chat as a `.txt` file.
- 🔁 **Regenerate Vectorstore**: Easily rebuild your document knowledge base.

---

## 🛠️ Tech Stack

| Tool          | Purpose                        |
|---------------|--------------------------------|
| Streamlit     | UI and App Framework           |
| LangChain     | Document Loading & Q&A Chain   |
| FAISS         | Vector Database for Embeddings |
| HuggingFace   | Sentence Embeddings            |
| OpenAI (GPT-4o) | LLM for answering queries     |
| Python        | Core language                  |

---

## 📁 Supported File Types

- `.pdf`
- `.docx`
- `.pptx`
- `.csv`
- `.txt`

---

## 📦 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/nuclearsh123/Notesbot-Chatbot.git
cd Notesbot-Chatbot
