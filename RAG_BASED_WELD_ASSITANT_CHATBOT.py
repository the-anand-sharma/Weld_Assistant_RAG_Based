import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import csv_loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import tempfile
from dotenv import load_dotenv
import os
 
load_dotenv()
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite",
                             google_api_key = os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Weld Assistant RAG Based")
st.title("Weld Assistant Powered By Gemini")

uploaded_file = st.file_uploader("Upload the file",type=["csv","txt"])

if uploaded_file :
    st.success("legend")
    with tempfile.NamedTemporaryFile(delete=False,suffix=".txt") as tf:
        tf.write(uploaded_file.getbuffer())
    uploaded_file_path = tf.name
    doc = TextLoader(uploaded_file_path,encoding="utf-8").load()
    
    with st.expander("Preview uploaded file"):
        st.write(doc[0].page_content[:1000])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(doc)

    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstores = FAISS.from_documents(chunks,embed)

    retriever = vectorstores.as_retriever(search_kwargs={"k":3})

    def format_docs(docss):
        return "\n\n".join([d.page_content for d in docss ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a welding expert. Answer the query based on this context:\n\n{context}"),
        ("human", "{question}")
    ])
    chain = {"context":retriever | format_docs, 'question':RunnablePassthrough()} | prompt | llm | StrOutputParser()

    question = st.text_input("Ask about your weld data:")
    if question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
        st.write(answer)


    

