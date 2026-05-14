import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup API Key
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# 2. UI and Branding
st.set_page_config(page_title="SAU AI Assistant", page_icon="🎓")
st.title("🎓 Sindh Agriculture University Digital Assistant")

# 3. Load the Text Database
try:
    with open("sau_data.txt", "r", encoding="utf-8") as file:
        sau_data = file.read()
except FileNotFoundError:
    sau_data = ""
    st.warning("Please upload 'sau_data.txt' to your GitHub repository.")

# 4. Initialize Groq (Ultra-fast Llama 3)
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful digital assistant for Sindh Agriculture University (SAU), Tando Jam. Use ONLY this official data to answer questions:\n\n{context}\n\nIf the answer is not in the data, say you don't know."),
    ("human", "{input}")
])

chain = prompt | llm

# 5. Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about SAU:"):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching SAU records..."):
                response = chain.invoke({"context": sau_data, "input": user_input})
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
        except Exception as e:
            st.error(f"Error connecting to the Groq server: {str(e)}")
