# ingestion.py

import getpass
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import cohere

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if not os.environ.get("PINECONE_API_KEY"):
  os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter Pinecone API key: ")

pinecone_api = os.environ["PINECONE_API_KEY"]
cohere_api = os.environ["COHERE_API_KEY"]

# vectorstore load
pc = Pinecone(api_key=pinecone_api)

index_name = "r50"
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)

retriever = vector_store.as_retriever(
  search_type="similarity", search_kwargs={"k": 10},
)

cohere_client = cohere.Client(cohere_api)