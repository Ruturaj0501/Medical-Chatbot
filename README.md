# Medical-Chatbot
Link-https://medical-chatbot-b9g3ctjb8rjstpnwmzgepe.streamlit.app/

# 🌿 RAG-Powered Wellness Assistant

This is a web application built with Streamlit that serves as a multi-functional wellness assistant. The application has two primary features:

1.  **Wellness Questionnaire:** A simple form that gathers information about a user's lifestyle, age, gender, and stress levels to calculate a wellness score. Based on this score, it provides a general supplement recommendation for Magnesium and L-Carnitine.
2.  **RAG-Powered Chatbot:** Users can upload one or more personal health documents (in PDF format). The application then uses a Retrieval-Augmented Generation (RAG) pipeline to answer user questions based *only* on the content of the uploaded documents. If no documents are uploaded, it acts as a general wellness chatbot.



## ✨ Features

-   **Interactive UI:** A clean and user-friendly interface built with Streamlit.
-   **Personalized Recommendations:** A questionnaire that provides a score-based supplement suggestion.
-   **Secure Document Handling:** Uploaded PDFs are processed in memory and are not stored permanently.
-   **Contextual Chat:** The RAG implementation ensures that the chatbot's answers are grounded in the user-provided documents, reducing hallucinations.
-   **Dual-Mode Chat:** Functions as a document-specific assistant when files are provided, and a general wellness assistant otherwise.

## 🛠️ Tech Stack

-   **Framework:** Streamlit
-   **LLM:** Google Gemini (`gemini-flash-latest`)
-   **Orchestration:** LangChain
-   **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
-   **Vector Store:** ChromaDB (in-memory)
-   **Document Loading:** PyPDFLoader
-   **Deployment (Example):** Streamlit Community Cloud, Hugging Face Spaces

## ⚙️ Setup and Installation

Follow these steps to run the application locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage project dependencies.

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
streamlit
google-generativeai
langchain-core
langchain-google-genai
langchain-chroma
langchain-text-splitters
langchain-community
langchain-huggingface
pypdf
sentence-transformers
chromadb
python-dotenv
```

Then, install them using pip:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a file named `.env` in the root directory of your project and add your API keys.

```env
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
HF_TOKEN="YOUR_HUGGINGFACE_HUB_TOKEN"
```

You can get your **Google API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey) and your **Hugging Face Token** from your [Hugging Face profile settings](https://huggingface.co/settings/tokens).

## 🚀 How to Run

Once you have completed the setup, run the following command in your terminal:

```bash
streamlit run app.py
```

This will start the Streamlit server, and the application will open in your default web browser.

## Usage

1.  **Fill the Questionnaire:** Complete the form on the main page and click "Get My Recommendation" to see your wellness score and a general suggestion.
2.  **Upload Documents:** Use the sidebar to upload one or more PDF files (e.g., medical reports, prescriptions).
3.  **Process Documents:** Click the "Process Documents" button. This will load, chunk, and embed the text into a vector store.
4.  **Ask Questions:** Use the chat input at the bottom of the page to ask questions.
    -   If you have processed documents, the chatbot will answer based on their content.
    -   If you haven't processed any documents, it will act as a general wellness assistant.

## ⚠️ Disclaimer

This application provides general suggestions and information. It is **not a substitute for professional medical advice**, diagnosis, or treatment. Always seek the advice of your physician or another qualified health provider with any questions you may have regarding a medical condition.
