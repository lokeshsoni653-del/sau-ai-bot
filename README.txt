# 🎓 Sindh Agriculture University (SAU) Digital Assistant

**Developed by:** * Lokesh Kumar (2K22-SE-42) 
* Vikram Sootahar (2K22-SE-36)

## Project Overview
The SAU Digital Assistant is an AI-powered conversational agent designed to help students, faculty, and visitors find accurate information regarding Sindh Agriculture University, Tando Jam. 

Built using a lightweight Retrieval-Augmented Generation (RAG) architecture, this assistant bypasses heavy vector databases in favor of a direct, high-speed text-filtering algorithm. This ensures maximum stability, eliminates external database dependency errors, and provides instant responses.

## Key Features
* **Custom Data Filtering:** Implements a custom Python algorithm to chunk and filter relevant paragraphs from the main dataset, efficiently managing token limits without sacrificing context.
* **Ultra-Fast LLM:** Powered by Groq's Llama 3.1 (8B) model for lightning-fast inference.
* **Knowledge Grounding:** Strictly bound to the official `sau_data.txt` knowledge base to prevent AI hallucinations.
* **Cloud Deployed:** Fully hosted and deployed via Streamlit Cloud.

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit
* **AI/LLM Framework:** LangChain (`langchain-groq`, `langchain-core`)
* **LLM Provider:** Groq API (Llama-3.1-8b-instant)
* **Language:** Python 3.11

## Project Structure
* `app.py` - The main application script containing the UI, filter logic, and LLM chain.
* `sau_data.txt` - The central knowledge base containing all official university information.
* `requirements.txt` - The list of Python dependencies required to run the app.

## How to Run Locally
If you want to run this project on your local machine, follow these steps:

1. Clone the repository:
git clone https://github.com/your-username/sau-ai-bot.git
cd sau-ai-bot

2. Install dependencies:
pip install -r requirements.txt

3. Set up your API Key:
Create a `.streamlit` folder in the root directory, and inside it, create a file named `secrets.toml`. Add your Groq API key:
GROQ_API_KEY = "your_api_key_here"

4. Run the application:
streamlit run app.py

## Future Enhancements
* Integration of OCR for automatic PDF prospectus reading.
* Expanding the `sau_data.txt` database with department-specific course outlines.
* Adding multi-language support (Sindhi, Urdu) for rural accessibility.
