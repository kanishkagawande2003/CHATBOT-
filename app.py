import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Page Configuration
st.set_page_config(page_title="Kanishka77-Multi-Doc Analyst", page_icon="📚", layout="wide")
st.title("🛡️ Kanishka77 Comparative Intelligence Engine")
st.markdown("---")

# 2. Sidebar Setup
with st.sidebar:
    st.header("Control Center")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    if st.button("Clear Knowledge Base"):
        st.session_state.clear()
        st.rerun()

# 3. Enhanced Comparison Prompt
template = """You are a Senior Strategic Analyst at P99Soft. 
You are analyzing MULTIPLE documents at once. 
1. Compare the data across different files if applicable.
2. If the user asks for a comparison, ALWAYS use a Markdown Table.
3. Highlight 'Gaps' (what one company does better than the other).

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 4. Main Application Engine
if groq_api_key:
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # FIX: Added accept_multiple_files=True
        uploaded_files = st.file_uploader("Upload all reports for comparison (PDF)", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            if "vectorstore" not in st.session_state:
                with st.spinner(f"🚀 Analyzing {len(uploaded_files)} documents..."):
                    all_docs = []
                    for uploaded_file in uploaded_files:
                        # Save each file temporarily
                        temp_filename = f"temp_{uploaded_file.name}"
                        with open(temp_filename, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        loader = PyPDFLoader(temp_filename)
                        all_docs.extend(loader.load())
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(all_docs)
                    
                    st.session_state.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                    st.success(f"Success! {len(uploaded_files)} reports merged into Knowledge Base.")

            # RAG Chain
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 8}) # Increased k for multi-doc

            def format_docs(docs):
                return "\n\n".join(f"[From {doc.metadata['source']}]: " + doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )

            # Chat UI
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Ask a comparative question..."):
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Comparing documents..."):
                        response = rag_chain.invoke(user_query)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"⚠️ System Error: {e}")
else:
    st.warning("Please enter your Groq API Key.")