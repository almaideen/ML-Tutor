import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader


from dotenv import load_dotenv
load_dotenv()

#Langsmith Tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']="ML Tutor"

#load the GROQ API KEY
groq_api_key=os.getenv('GROQ_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

loader = PyPDFDirectoryLoader('Data') #Data Ingestion
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=500)
final_documents = text_splitter.split_documents(docs)

llm = ChatGroq(groq_api_key=groq_api_key,model_name='llama-3.1-70b-versatile')
embeddings=OpenAIEmbeddings()

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on thr provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question:{input}
    """
)
vectorstore = FAISS.from_documents(final_documents,embeddings)
vectorstore.save_local("faiss_index")

st.title("ML TUTOR")

user_prompt=st.text_input("Enter your question")

if user_prompt:
    faiss_vectors=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=faiss_vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response=retrieval_chain.invoke({'input':user_prompt})
    st.write(response['answer'])