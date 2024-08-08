import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os, getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


#Gemini Key
os.environ['GOOGLE_API_KEY']="AIzaSyB6-jZLBXeOeLFBhFaU11oidwAeBATkrds"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    #RecursiveCharacterTextSplitter CharacterTextSplitter separator="\n", 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000, length_function=len)#
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain(Fvs):

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=model,retriever=Fvs.as_retriever(), memory=memory)

    return chain



def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("Bot: ", message.content)

## streamlit app
st.set_page_config("Chat With Multiple PDF")
st.header("Chat with Multiple PDF using Gemini :books:")

user_question = st.text_input("Ask a Question from the PDF Files")
submit=st.button("Ask the question")

## If ask button is clicked
if submit:
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
         with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            Fvs = get_vector_store(text_chunks)
            st.session_state.conversation = get_conversational_chain(Fvs)
            st.success("Done")

footer = """
---
#### Made By [Surat Banerjee](https://www.linkedin.com/in/surat-banerjee/)
For Any Queries, Reach out on [Portfolio](https://suratbanerjee.wixsite.com/myportfoliods)  
"""

st.markdown(footer, unsafe_allow_html=True)
