import streamlit as st
import time
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

# 2. Setup the AI Engine & Database (Modern LCEL Architecture)
@st.cache_resource
def initialize_ai():
    # Load the local database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./sau_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4}) 
    
    # Connect to the free Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key="AIzaSyABemVGFRtOHMFFmM0PeX3UnHd0zbk3OiU", # <--- PASTE YOUR API KEY HERE
        temperature=0.3
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
    
    # Helper function to format the retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # The Modern LCEL Chain (Bypasses the broken 'langchain.chains' module entirely!)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = initialize_ai()

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
            # Invoke the chain directly with the user's string
            answer = rag_chain.invoke(prompt)
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})