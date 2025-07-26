import os
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_mongodb import MongoDBAtlasVectorSearch

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup MongoDB Connection
client = MongoClient(MONGO_URI)
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print("MongoDB Connection Error:", e)
    exit()

# Configure Database and Collection
db = client["test_db"] 
collection = db["vector_store"]  
VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Load PDF Document
files = ["files/AssetLens_Complete_User_Guide.pdf"]

# Load all PDFs and combine their data
datas = []
for file in files:
    loader = PyPDFLoader(file)
    datas.extend(loader.load())

# Split Document into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=200)
chunks = text_splitter.split_documents(datas)

# Generate Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store Embeddings in MongoDB Atlas Vector Search
vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    collection=collection,
    index_name=VECTOR_SEARCH_INDEX_NAME
)

# Create a Retriever
retriever = vector_search.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define Prompt Template
template = """You are an assistant for the AssetLens app. Answer the question below using only the provided context from the AssetLens documentation. 
If the question is not related to AssetLens or cannot be answered using the context, politely respond: "Sorry, I can only answer questions related to AssetLens."

Question: {question}
Context: {context}
Answer:"""
prompt = PromptTemplate.from_template(template)

# Configure Chat Model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.5, api_key=GEMINI_API_KEY)

# Define RAG Chain
retrieval_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

# Continuous Chat Loop
def chat():
    print("Ask your questions (type 'exit' to quit):")
    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            print("Chat closed.")
            break
        response = retrieval_chain.invoke(query)
        print("AI:", response)

if __name__ == "__main__":
    chat()