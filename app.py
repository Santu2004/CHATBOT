import streamlit as st
import os
import uuid
import hashlib

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.db.sqlite import SqliteDb

from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.openweather import OpenWeatherTools

from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.vectordb.chroma import ChromaDb


# =========================
# Load .env
# =========================

load_dotenv()


# =========================
# Dynamic session ID
# =========================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# =========================
# Chat history for UI
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# Page
# =========================

st.title("🤖 CHATBOT OF SANTUUU")

st.caption(
    "Agno + Groq + Tools + Memory + PDF RAG"
)


# =========================
# SQLite Memory
# =========================

db = SqliteDb(
    db_file="agent.db"
)


# =========================
# ChromaDB
# =========================

vector_db = ChromaDb(
    collection="my_pdf_knowledge",
    path="chroma_data",
    persistent_client=True,
    embedder=FastEmbedEmbedder()
)


# =========================
# Knowledge
# =========================

knowledge = Knowledge(
    vector_db=vector_db
)


# =========================
# PDF Upload
# =========================

st.sidebar.header("📄 PDF")

pdf = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


if pdf:

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    pdf_path = os.path.join(
        "uploads",
        pdf.name
    )

    pdf_bytes = pdf.getvalue()

    pdf_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()


    # Save PDF
    with open(
        pdf_path,
        "wb"
    ) as file:

        file.write(pdf_bytes)


    st.sidebar.success(
        "PDF uploaded!"
    )


    # Add PDF button
    if st.sidebar.button(
        "📥 Add PDF to Agent"
    ):

        if st.session_state.get(
            "pdf_hash"
        ) != pdf_hash:

            with st.spinner(
                "Reading PDF..."
            ):

                knowledge.insert(
                    path=pdf_path,
                    skip_if_exists=True
                )


            st.session_state.pdf_hash = pdf_hash

            st.sidebar.success(
                "✅ PDF added to knowledge!"
            )

        else:

            st.sidebar.info(
                "PDF is already added."
            )


# =========================
# Agent
# =========================

agent = Agent(

    model=Groq(
        id="openai/gpt-oss-20b"
    ),

    db=db,

    knowledge=knowledge,

    search_knowledge=True,

    tools=[

        CalculatorTools(),

        DuckDuckGoTools(),

        OpenWeatherTools(
            enable_current_weather=True,
            enable_forecast=True,
            enable_geocoding=True,
            units="metric"
        )
    ],

    add_history_to_context=True,

    num_history_runs=10,

    update_memory_on_run=True,

    instructions="""

    You are a helpful AI assistant.

    Use the calculator for calculations.

    Use DuckDuckGo for current information.

    Use the weather tool for weather questions.

    Use the PDF knowledge base for questions
    about the uploaded PDF.

    Answer clearly and simply.

    """,

    markdown=True
)


# =========================
# Show previous messages
# =========================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================
# Chat Input
# =========================

question = st.chat_input(
    "Ask me anything..."
)


# =========================
# New Question
# =========================

if question:

    # -------------------------
    # Show user question
    # -------------------------

    st.chat_message(
        "user"
    ).markdown(question)


    # Save user question
    st.session_state.messages.append({

        "role": "user",

        "content": question
    })


    # -------------------------
    # Stream Agent Response
    # -------------------------

    with st.chat_message(
        "assistant"
    ):

        response = agent.run(

            question,

            user_id="user_1",

            session_id=st.session_state.session_id,

            stream=True
        )


        # Convert Agno stream into text
        def generate_response():

            for chunk in response:

                if chunk.content:

                    yield chunk.content


        # Display streaming response
        answer = st.write_stream(
            generate_response()
        )


    # -------------------------
    # Save complete answer
    # -------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer
    })


# =========================
# Clear Chat
# =========================

if st.sidebar.button(
    "🗑️ Clear Chat"
):

    st.session_state.messages = []

    st.session_state.session_id = str(
        uuid.uuid4()
    )

    st.rerun()