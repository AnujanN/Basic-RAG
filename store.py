import os
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("DB_URL")

# Setup MongoDB Connection
client = MongoClient(MONGO_URI)
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB!")
except Exception as e:
    print("MongoDB Connection Error:", e)
    exit()

# Configure Database and Collection
db = client["RAG"]
collection = db["vector_store"]
VECTOR_SEARCH_INDEX_NAME = "vector_index"

# List of PDF files
files = ["files/AssetLens_Complete_User_Guide.pdf"]

# Load all PDFs and combine their data
documents = []
for file in files:
    loader = PyPDFLoader(file)
    documents.extend(loader.load())

# Split Documents into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Generate Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store Embeddings in MongoDB Atlas Vector Search
vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection=collection,
    index_name=VECTOR_SEARCH_INDEX_NAME
)

print("Embeddings stored successfully!")
