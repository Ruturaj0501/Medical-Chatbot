import streamlit as st
import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={"device": "cpu"})
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.7)

st.set_page_config(page_title="Wellness Assistant", page_icon="💊")
st.title("🌿 RAG-Powered Wellness Assistant")
st.write(
    "Hello! Please fill out the questionnaire below. You can also upload one or more health documents (like prescriptions) "
    "and ask me questions about them in the chat box."
)


def get_recommendation(total_score):
    if total_score >= 5:
        return (
            "**Risk Category:** High\n\n"
            "**Suggestion:** Based on your responses, we recommend **MGT Plus capsule** for **6 months**, taking one capsule every day."
        )
    elif 3 <= total_score <= 4:
        return (
            "**Risk Category:** Medium\n\n"
            "**Suggestion:** Based on your responses, we recommend **MGT Plus capsule** for **3 months**, taking one capsule every day."
        )
    elif 1 <= total_score <= 2:
        return (
            "**Risk Category:** Low\n\n"
            "**Suggestion:** Based on your responses, we recommend **MGT Capsule** for **3 months**, taking one capsule every day."
        )
    else:
        return (
             "**Risk Category:** Very Low\n\n"
             "**Suggestion:** Your lifestyle seems well-balanced! You may not require supplementation at this time."
        )

def calculate_scores(answers):
    mg_score = 0
    lc_score = 0
    if answers.get('gender') == 'Female': mg_score += 1
    if answers.get('age') == '46–60':
        mg_score += 1; lc_score += 1
    elif answers.get('age') == '60+':
        mg_score += 2; lc_score += 2
    work_type = answers.get('work_type')
    if work_type in ['Desk', 'Physical']: lc_score += 1
    elif work_type in ['Homemaker', 'Shift']: mg_score += 1
    elif work_type == 'Retired':
        mg_score += 1; lc_score += 1
    sleep_stress = answers.get('sleep_stress')
    if sleep_stress == '5–6 hours, occasional stress': mg_score += 1
    elif sleep_stress == 'Less than 5 hours, often stressed':
        mg_score += 2; lc_score += 1
    elif sleep_stress == 'Night cramps, restless legs, very tired':
        mg_score += 2; lc_score += 2
    return {'mg': mg_score, 'lc': lc_score}


if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "scores" not in st.session_state:
    st.session_state.scores = None

def process_pdfs(_uploaded_files):
    with st.spinner("Processing your documents... This may take a moment."):
        try:
            documents = []
            for file in _uploaded_files:
                temp_path = f"./{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                
                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())
                os.remove(temp_path)
            
            if not documents:
                st.warning("Could not extract any text from the uploaded PDF(s). This can happen if the file is a scanned image or is empty. Please upload a text-based PDF.")
                return None
           
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.success("PDFs processed and ready!")
            return vectorstore.as_retriever()
        
        except Exception as e:
            st.error(f"An error occurred while processing the PDFs: {e}")
            return None
        
with st.sidebar:
    st.header("📄 Document Upload for RAG")
    uploaded_files = st.file_uploader(
        "Upload your prescription(s) or health report(s)",
        type="pdf",
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Process Documents"):
            st.session_state.retriever = process_pdfs(uploaded_files)


with st.form("wellness_form"):
    st.header("Wellness Questionnaire")
    st.subheader("Part 1: About You")
    gender = st.selectbox("Gender:", ["Male", "Female", "Other"], key="gender")
    age = st.selectbox("Age:", ["18–45", "46–60", "60+"], key="age")
    work_type = st.selectbox("Work type:", ["Desk", "Physical", "Homemaker", "Retired", "Shift"], key="work_type")
    st.divider()
    st.subheader("Part 2: Sleep and Stress")
    sleep_stress = st.radio(
        "How would you describe your sleep and stress levels?",
        ["Restful 7–8 hours, rarely stressed", "5–6 hours, occasional stress", "Less than 5 hours, often stressed", "Night cramps, restless legs, very tired"],
        key="sleep_stress"
    )
    if st.form_submit_button("Get My Recommendation"):
        st.session_state.scores = calculate_scores({'gender': gender, 'age': age, 'work_type': work_type, 'sleep_stress': sleep_stress})


if st.session_state.scores:
    st.header("Your Wellness Profile")
    mg, lc = st.session_state.scores['mg'], st.session_state.scores['lc']
    total = mg + lc
    col1, col2, col3 = st.columns(3)
    col1.metric("Magnesium Score", mg); col2.metric("L-Carnitine Score", lc); col3.metric("Total Score", total)
    st.info(get_recommendation(total))
    st.warning("**Disclaimer:** This is a general suggestion, not a medical prescription. Consult a healthcare provider before starting any new supplement.")

st.divider()

st.header("Ask Me Anything")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about your document(s), or general wellness questions..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ""
           
            if st.session_state.retriever:
                retriever = st.session_state.retriever
                template = """
                You are a helpful wellness assistant. Answer the question based ONLY on the following context from the user's uploaded document(s).
                If the answer is not in the context, state that you don't have enough information from the document(s).
                Do not provide medical advice. Always suggest consulting a healthcare professional.

                Context:
                {context}

                Question:
                {question}
                """
                prompt = PromptTemplate.from_template(template)
                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                response = rag_chain.invoke(user_query)
          
            else:
                template = """
                You are a helpful wellness assistant. Your role is to provide supportive and informative answers related to Magnesium (Mg), L-Carnitine (LC), and general well-being.
                If you are asked about a document, politely state that no document has been uploaded and processed yet.
                Do not provide medical advice. Always suggest consulting a healthcare professional.

                Question: {question}
                """
                prompt = PromptTemplate.from_template(template)
                general_chain = prompt | llm | StrOutputParser()
                response = general_chain.invoke({"question": user_query})
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})