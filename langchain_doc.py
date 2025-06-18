from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7,
    max_output_tokens=1024
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

def create_vector_db_from_youtube_url(youtube_url:str)->FAISS:
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(query:str, db:FAISS, k:int=4)->str:
    docs = db.similarity_search(query, k=k)
    docs_page_content = "\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant that can answer questions about the following text:
        {context}
        Question: {question}
        Answer:

        only use factual information from the text. to answer the question. if you don't know the answer, just say "I don't know"

        your answers should be detailed and comprehensive.
        """,
        input_variables=["question", "context"]
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": query, "context": docs_page_content})
    return response