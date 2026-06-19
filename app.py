from dotenv import load_dotenv
load_dotenv()

import os
import json
import shutil
import re
import pandas as pd
import streamlit as st

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.utilities import GoogleSerperAPIWrapper

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "llm" not in st.session_state:
    st.session_state.llm = None

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Fact Check Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fact Check Agent")

# --------------------------------------------------
# FACT CHECK REPORT
# --------------------------------------------------

def generate_fact_check_report():
    vector_db = st.session_state.vector_db
    llm = st.session_state.llm

    if vector_db is None or llm is None:
        return pd.DataFrame([{"Claim": "System Error: Missing DB or LLM", "Status": "ERROR", "Correct Fact": ""}])

    # 1. Retrieve context
    docs = vector_db.similarity_search(
        query="statistics percentages dates revenue market size metrics claims",
        k=20
    )
    document_text = "\n\n".join([doc.page_content for doc in docs])

    # 2. Extract Claims
    extract_prompt = f"""
Extract verifiable factual claims from the document.
Focus only on: statistics, percentages, dates, revenue numbers, market sizes, technical metrics.

Return ONLY a valid JSON array of objects. Do not include markdown formatting or extra text.
Example: [{{"claim":"OpenAI was founded in 2015"}}]

Document text:
{document_text}
"""
    extraction = llm.invoke(extract_prompt)

    try:
        # Regex to force-find the JSON array, ignoring any conversational hallucinations
        match = re.search(r'\[.*\]', extraction.content, re.DOTALL)
        if match:
            claims = json.loads(match.group(0))
        else:
            claims = json.loads(extraction.content.replace("```json", "").replace("```", ""))
    except Exception as e:
        return pd.DataFrame([{"Claim": "Failed to parse claims from LLM.", "Status": "ERROR", "Correct Fact": str(e)}])

    # 3. Verify Claims
    serper = GoogleSerperAPIWrapper()
    rows = []

    for item in claims:
        claim = item.get("claim", "")
        if not claim:
            continue

        try:
            evidence = serper.run(claim)

            verify_prompt = f"""
Claim: "{claim}"
Web Evidence: {evidence}

Determine if the claim is VERIFIED, INACCURATE, or FALSE based on the evidence.
Use these STRICT definitions:
- VERIFIED: The claim is completely true.
- INACCURATE: The core subject exists, but specific numbers, dates, or details are wrong.
- FALSE: The claim is entirely made up.

Return ONLY a valid JSON object. Do not include markdown.
Example: {{"status":"INACCURATE", "correct_fact":"The actual number is X."}}
"""
            verification = llm.invoke(verify_prompt)

            # Regex to force-find the JSON object
            match = re.search(r'\{.*\}', verification.content, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = json.loads(verification.content.replace("```json", "").replace("```", ""))

            rows.append({
                "Claim": claim,
                "Status": parsed.get("status", "ERROR").upper(),
                "Correct Fact": parsed.get("correct_fact", "No correction provided.")
            })

        except Exception as e:
            rows.append({"Claim": claim, "Status": "ERROR", "Correct Fact": f"Verification failed."})

    return pd.DataFrame(rows)

# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

def process_document(folder_path):
    uploaded_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    if not uploaded_files:
        st.error("No PDF found")
        return

    target_file_path = os.path.join(folder_path, uploaded_files[0])

    # Load and Split
    loader = PyMuPDF4LLMLoader(target_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings and DB
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)

    st.session_state.vector_db = vector_db

    # Tools
    serper = GoogleSerperAPIWrapper()

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant content from uploaded PDF."""
        docs = vector_db.similarity_search(query=query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])

    @tool
    def web_search(query: str) -> str:
        """Search live web for fact checking."""
        return serper.run(query)

    # LLM and Agent
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    st.session_state.llm = llm

    system_prompt = """
You are an AI assistant strictly for PDF analysis and fact checking.

CRITICAL CONTEXT: 
A PDF document has ALREADY been uploaded, processed, and is ready for you to analyze. Do not ask the user to provide a document. 

You have access to:
1. retrieve_context (Searches the uploaded PDF)
2. web_search (Searches the live web)

RULES:
- STRICT SCOPE GUARDRAIL: You are exclusively a document analysis and fact-checking tool. If the user asks a question that is entirely unrelated to the uploaded document or verifying a claim (e.g., general knowledge, math like "1+1", coding, casual chat), you MUST politely refuse to answer. Reply exactly with: "I am specifically designed to analyze the uploaded document and fact-check claims. Please ask a question related to the PDF."
- If the user asks about the document (summarize, key points, explain), use retrieve_context and answer ONLY from the PDF.
- If the user explicitly asks to "fact check" or "verify" a specific claim, use retrieve_context to find the claim, then use web_search to verify it.

When returning a VERDICT, you MUST use these strict definitions:
- VERIFIED: The claim is completely true.
- INACCURATE: The core subject exists, but specific numbers, dates, or details are wrong.
- FALSE: The claim is entirely made up.

FORMATTING RULES:
You MUST format your response using clear markdown and line breaks. Use this exact structure:

**VERDICT:** [VERIFIED / INACCURATE / FALSE]

**EXPLANATION:**
[Explain your reasoning clearly, using bullet points if there are multiple claims.]

**CORRECT FACT:**
[Provide updated information if available.]
"""

    memory = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=[retrieve_context, web_search],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

# --------------------------------------------------
# UPLOAD SCREEN
# --------------------------------------------------

if not st.session_state.document_uploaded:
    uploaded_files = st.file_uploader("Upload PDF Document", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Processing document..."):
            folder_path = "./doc_files"
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

            for file in uploaded_files:
                file_path = os.path.join(folder_path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            # 1. Process the document
            process_document(folder_path)

            # 2. Immediately trigger the automated report
            with st.spinner("Automatically extracting and verifying claims..."):
                report_df = generate_fact_check_report()
                
                # 3. Inject the report as the very first message in the chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hello! I have automatically analyzed the uploaded document and verified its core claims. Here is your fact-check report. You can also ask me specific questions about the document below.",
                    "df": report_df
                })

        st.success("Document analyzed successfully.")
        st.rerun()

# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

if st.session_state.document_uploaded and st.session_state.agent:

    st.sidebar.title("Options")

    if st.sidebar.button("Clear Chat History & Upload New PDF"):
        st.session_state.messages = []
        st.session_state.agent = None
        st.session_state.document_uploaded = False
        st.session_state.vector_db = None
        st.session_state.llm = None
        if os.path.exists("./doc_files"):
            shutil.rmtree("./doc_files")
        st.rerun()

    # Inject the welcome message if the chat is completely empty
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello. How can I assist you with the uploaded PDF document? Would you like me to verify the claims/fact check, or explain a specific part of it?"
        })

    # Chat History Rendering Loop (Includes DataFrame handling)
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If the message contains a dataframe, render it permanently
            if "df" in message and isinstance(message["df"], pd.DataFrame):
                st.dataframe(message["df"], use_container_width=True)
                csv = message["df"].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Report as CSV",
                    data=csv,
                    file_name=f"fact_check_report_{i}.csv",
                    mime="text/csv",
                    key=f"dl_btn_{i}"
                )

    query = st.chat_input("Ask about the PDF or type 'fact check' to generate a report...")

    if query:
        # 1. Show user query immediately
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # ---------------------------------------------------------
        # NEW SMART ROUTER (Replaces the hardcoded keyword list)
        # ---------------------------------------------------------
        router_prompt = f"""
        You are an intent classification routing AI. 
        Analyze this user message: "{query}"
        
        If the user is asking to broadly fact-check the document, verify all claims, generate a report, or create a table/CSV of claims, output the exact word: REPORT
        If the user is asking a specific question, asking to verify a single specific claim, asking for a summary, or just chatting casually, output the exact word: CHAT
        
        Return ONLY one word. No punctuation.
        """
        
        # Ask the LLM to classify the intent
        if st.session_state.llm:
            intent = st.session_state.llm.invoke(router_prompt).content.strip().upper()
        else:
            intent = "CHAT" # Fallback if the LLM somehow isn't loaded

        # 2. Automated Report Generation Logic (Triggered by the AI's classification)
        if "REPORT" in intent:
            with st.spinner("Analyzing document and cross-referencing claims with the web..."):
                report_df = generate_fact_check_report()

            # Append the report to messages so it never disappears
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Here is the automated fact-check report for the document:",
                "df": report_df
            })
            st.rerun()

        # 3. Standard RAG Chatbot Logic (Triggered by the AI's classification)
        else:
            with st.spinner("Thinking..."):
                response = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    {"configurable": {"thread_id": "fact_check_session"}}
                )
                answer = response["messages"][-1].content

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})