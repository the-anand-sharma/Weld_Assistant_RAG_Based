from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

text = TextLoader("weld_data.txt",'utf-8')
docs = text.load()

splitter = RecursiveCharacterTextSplitter(chunk_size= 200,chunk_overlap= 50,
                                          separators=["\n\n", "\n", " ", ""],)



chunks = splitter.split_documents(docs)

embed = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")
vectorstores = FAISS.from_documents(chunks,embed)
retriever = vectorstores.as_retriever(search_kwargs={"k":3})


def format_docs(docss):
    return "\n\n".join([d.page_content for d in docss ])

prompt = ChatPromptTemplate([("system","you are a welding expert answer the question according to this information{context}"),
                            ("human","{question}")])

chain = {"context":retriever | format_docs, 'question':RunnablePassthrough()} | prompt | llm | StrOutputParser()
result = chain.invoke("what are the welding parameters in WPS-2026-001?")
print(result)



