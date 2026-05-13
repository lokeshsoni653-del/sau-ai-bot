import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Page Setup
st.set_page_config(page_title="SAU AI Assistant", page_icon="🎓")
st.title("🎓 SAU AI Bot (Stable Version)")

# 2. Load Local Database
@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./sau_db", embedding_function=embeddings)
    return db

db = load_knowledge_base()

# 3. Setup Google's AI Brain
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # Using 'gemini-pro' for maximum stability
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"API Setup Error: {e}")

# 4. Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about SAU Admissions, Fees, or Departments..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Search local database for facts
        with st.spinner("Searching SAU Database..."):
            docs = db.similarity_search(prompt, k=4)
            context = "\n".join([d.page_content for d in docs])
        
        # Send facts to Gemini
        with st.spinner("AI is thinking..."):
            system_prompt = f"""
            You are the official SAU AI Assistant. 
            Answer the user's question using ONLY the following context:
            {context}
            
            If the answer isn't in the context, say 'I don't have that specific information in my database.'
            """
            try:
                response = model.generate_content(f"{system_prompt}\n\nUser Question: {prompt}")
                answer = response.text
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("The AI Brain is currently offline. Showing raw data from database instead:")
                st.warning(context)
