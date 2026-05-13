import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Branding
st.set_page_config(page_title="SAU AI Assistant", page_icon="🎓")
st.title("🎓 SAU AI Assistant")

# 2. Load Local Database (The "Textbook")
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="./sau_db", embedding_function=embeddings)

db = load_db()

# 3. Setup AI Brain (The "Student")
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # We use 'gemini-1.5-flash' - the newest and fastest
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Setup Error: {e}")

# 4. Chat logic
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about SAU..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Search the local database for SAU facts
        docs = db.similarity_search(prompt, k=4)
        context = "\n".join([d.page_content for d in docs])
        
        # Build the final prompt
        full_query = f"Using this info:\n{context}\n\nAnswer this: {prompt}"
        
        try:
            response = model.generate_content(full_query)
            answer = response.text
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error("AI Brain Offline. Here is the data from the SAU Database:")
            st.info(context) # This shows the raw SAU data so you can still present!
