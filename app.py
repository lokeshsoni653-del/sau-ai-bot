import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Config & Branding
st.set_page_config(page_title="SAU AI Assistant", page_icon="🎓", layout="centered")
st.image("https://sau.edu.pk/wp-content/uploads/2023/10/SAU-Logo-1.png", width=150)
st.title("🎓 Sindh Agriculture University AI Bot")
st.markdown("Ask me anything about admissions, departments, or life at SAU Tandojam!")

# 2. Setup the AI Engine & Database
@st.cache_resource
def initialize_ai():
    # Force the API key into the system environment
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    
    # Load the local database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./sau_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4}) 
    
    # Connect to the Gemini LLM
    # NOTE: If gemini-1.5-flash still gives a 404, change it to "gemini-pro"
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        convert_system_message_to_human=True
    )
    
    # Give the AI its personality
    template = """You are the official AI Assistant for Sindh Agriculture University (SAU), Tando Jam.
    Use the following retrieved context to answer the user's question accurately. 
    If the answer is not in the context, politely say that you don't have that information.
    Keep your answers helpful, professional, and concise.

    Context:
    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # The Modern LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Initialize the chain
try:
    rag_chain = initialize_ai()
except Exception as e:
    st.error(f"Error initializing AI: {e}")
    st.stop()

# 3. Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture User Input
if prompt := st.chat_input("E.g., What are the criteria for BS Software Engineering?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Searching SAU database..."):
            try:
                answer = rag_chain.invoke(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("I'm having trouble connecting to the AI server. Please try again in a moment.")
                st.info(f"Technical details: {e}")
