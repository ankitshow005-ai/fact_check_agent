---

# 🔍 Automated Fact-Check Agent

A smart, dual-purpose web application built with Streamlit and LangChain that acts as a "Truth Layer" for your documents. Simply upload a PDF, and the AI will automatically extract major claims, cross-reference them with live web data, and generate an instant fact-check report.

**[🔗 Click here to try the Live Demo!](https://myfactcheckagent.streamlit.app/)**

---

## ✨ Features

* **⚡ Automated Claim Verification:** The moment you upload a PDF, the app extracts verifiable statistics, dates, and figures, and automatically grades them as `VERIFIED`, `INACCURATE`, or `FALSE`.
* **🌐 Live Web Cross-Referencing:** Powered by the Google Serper API, the agent searches the live internet to find the real, up-to-date facts to contrast against the document.
* **📊 CSV Export:** Instantly download your automated fact-check reports as a CSV file.
* **🤖 Smart Conversational RAG Chatbot:** Ask follow-up questions about the document, or ask the bot to verify new, specific claims on the fly.
* **🧠 Intelligent Intent Routing:** No need for clunky slash-commands. The app features an LLM Semantic Router that automatically detects whether you want to chat casually or generate a new data table.

---

## 🛠️ How It Works

1. **Upload:** You upload a PDF document containing marketing claims, statistics, or reports.
2. **Process:** The document is chunked, embedded using Google GenAI, and stored in an In-Memory Vector Database.
3. **Automated Report:** A dedicated LLM prompt extracts the most verifiable claims. It then iterates through them, using a web search tool to find the truth, and outputs a formatted Data Table.
4. **Follow-Up Chat:** An AI Agent (powered by Groq and Llama-3) equipped with `retrieve_context` and `web_search` tools sits ready to answer any specific questions you have about the text.

---

## 🚀 Run It Locally

If you want to run this app on your own machine, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/ankitshow005-ai/fact_check_agent.git
cd fact_check_agent

```

### 2. Install the requirements

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt

```

### 3. Set up your API Keys

Create a file named `.env` in the root folder of the project and add your API keys:

```env
GROQ_API_KEY="your_groq_api_key"
GOOGLE_API_KEY="your_google_api_key"
SERPER_API_KEY="your_serper_api_key"

```

*(Note: Never commit your `.env` file to GitHub!)*

### 4. Run the app

```bash
streamlit run app.py

```

---

## 💡 Usage Guide

* **The Trap Test:** Upload a document containing intentional lies or outdated statistics. Watch as the app instantly generates a table calling out the falsehoods and providing the corrected data.
* **Natural Language Routing:** In the chat box, you can type things like *"verify all claims"* or *"hit me with a report"* and the app will smartly generate a new CSV table. If you type *"What is this document about?"*, it will converse normally.

---

### Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain & LangGraph
* **LLM:** Llama-3 (via Groq)
* **Embeddings:** Google Generative AI (`gemini-embedding-2-preview`)
* **Web Search:** Google Serper API
* **PDF Parsing:** PyMuPDF4LLM