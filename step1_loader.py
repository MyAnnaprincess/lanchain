from langchain_community.document_loaders import PyPDFLoader

from langchain_zhipu import ZhipuAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = ZhipuAIEmbeddings()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


loader = PyPDFLoader('chatGPT调研报告.pdf')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)

docs = loader.load_and_split(text_splitter)
print(docs)
print(len(docs))

chroma = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

chroma.add_documents(docs)
print("uploaded")
