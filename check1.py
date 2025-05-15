import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.utilities import SerpAPIWrapper
import google.generativeai as genai

# Load environment variables
load_dotenv()
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize tools
search = SerpAPIWrapper()
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# PDF text extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Text chunking
def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return splitter.split_text(text)

# Embedding & vector store
def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

# Conversational chain
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# Main app
def main():
    st.set_page_config(page_title="AI PDF + Web Chat", layout="wide")
    st.title("🤖 AI Chat Assistant")

    menu = ["📄 Upload PDF", "💬 Chat with Document", "🌐 Ask AI with Web"]
    choice = st.sidebar.radio("Choose Mode", menu)

    if "chat_history_doc" not in st.session_state:
        st.session_state.chat_history_doc = []
    if "chat_history_web" not in st.session_state:
        st.session_state.chat_history_web = []

    if choice == "📄 Upload PDF":
        st.subheader("📄 Upload and Process PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Extracting and indexing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ PDFs processed! Now go to '💬 Chat with Document'.")

    elif choice == "💬 Chat with Document":
        st.subheader("💬 Chat with Your PDFs")

        if "conversation" not in st.session_state:
            st.warning("Please upload and process PDFs first.")
        else:
            for msg in st.session_state.chat_history_doc:
                with st.chat_message("user"):
                    st.markdown(msg["user"])
                with st.chat_message("assistant"):
                    st.markdown(msg["bot"])

            user_input = st.chat_input("Ask something about your PDFs...")
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("🤖 Thinking..."):
                        try:
                            full_prompt = f"Answer step-by-step: {user_input}"
                            response = st.session_state.conversation({"question": full_prompt})
                            answer = response['answer'].strip()

                            # Fallback if no good answer
                            if not answer or "I'm sorry" in answer or "does not contain" in answer:
                                serp_result = search.run(user_input)
                                web_prompt = f"""You are a helpful AI tutor. Based on the real-time web result and your own knowledge, answer the following question step-by-step.

Question: {user_input}
Web Search Result: {serp_result}

Answer:"""
                                gemini_response = gemini_model.generate_content(web_prompt)
                                answer = f"📡 *Web Fallback*:\n\n{gemini_response.text.strip()}"

                            st.markdown(answer)
                            st.session_state.chat_history_doc.append({"user": user_input, "bot": answer})
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history_doc = []

    elif choice == "🌐 Ask AI with Web":
        st.subheader("🌐 Ask Anything (with Web Support) 🔍")

        for msg in st.session_state.chat_history_web:
            with st.chat_message("user"):
                st.markdown(msg["user"])
            with st.chat_message("assistant"):
                st.markdown(msg["bot"])

        user_question = st.chat_input("Type your question... (e.g. Who won the IPL 2025?)")
        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("🌐 Searching the web and generating answer..."):
                    try:
                        serp_result = search.run(user_question)
                        prompt = f"""You are a helpful AI tutor. Based on the real-time web result and your own knowledge, answer the following question step-by-step.

Question: {user_question}
Web Search Result: {serp_result}

Answer:"""
                        response = gemini_model.generate_content(prompt)
                        final_answer = response.text.strip()
                        st.markdown(final_answer)
                        st.session_state.chat_history_web.append({"user": user_question, "bot": final_answer})
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            if st.button("🗑️ Clear Web Chat History"):
                st.session_state.chat_history_web = []

if __name__ == '__main__':
    main()
