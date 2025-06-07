# 📘 Contextual-Retrival Based RAG Chatbot

This is a Python-based chatbot designed to assist uses technique of Contextual Retrieval along with RAG system to optimized the LLM response and make it more accurate. It uses LangChain, HuggingFace embeddings, Chroma for vector storage, and Groq's LLMs (e.g., LLaMA 3) to generate helpful responses based on document context.

---

## 📰 Read the Full Walkthrough

📖 **Medium Article**: [Building an Institutional Chatbot with PDF, LangChain, and Groq](https://medium.com/your-article-link-here)
(*Replace this with your actual Medium article link*)

---

## 🚀 Features

* Parses PDF files to extract structured information
* Classifies content into categories (e.g., geography, politics, education)
* Uses embedding-based search for question answering
* Integrates with Groq LLM for natural language response generation
* Categorized metadata for intelligent context generation

---

## 📁 Project Structure

```
.
├── data/
│   └── knowledge_base.pdf     # Your institutional PDF document
├── vector_database/           # Stores vector embeddings (auto-generated)
├── .env                       # Stores environment variables
├── main.py                 # Main chatbot application
└── README.md                  # This file
```

---

## ⚙️ Requirements

* Python 3.8 or higher
* pip (Python package manager)

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/institutional-chatbot.git
cd institutional-chatbot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

Create `requirements.txt` with:

```txt
python-dotenv
langchain
langchain-community
langchain-chroma
langchain-huggingface
sentence-transformers
groq
```

3. **Add your `.env` file**

Create a `.env` file in the root directory and include your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📥 Add Your PDF

Place your institutional document inside the `data/` folder and name it `knowledge_base.pdf`.

You can change the file name and path in the code if needed.

---

## ▶️ Running the Chatbot

Run the chatbot using:

```bash
python main.py
```

---

## 🗨️ Usage

* Type your institutional question and press Enter.
* Type `help` to see available commands.
* Type `clear` to clear the terminal screen.
* Type `quit`, `exit`, or `bye` to end the session.

---

## 🛠️ Customization

* You can extend `self.content_categories` in `Chatbot.__init__()` to handle more domains.
* The embedding model and LLM can be swapped in the class constructor.

---

## 🧠 Notes

* Ensure your PDF has extractable text content.
* The vector store is persistent and will be reused unless deleted manually.
* Groq’s LLM is used for high-quality response generation.
