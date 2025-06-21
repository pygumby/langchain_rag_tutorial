import getpass
import os

# Chat model
from langchain.chat_models import init_chat_model

# Embeddings model
from langchain_openai import OpenAIEmbeddings

# Vector store
from langchain_core.vectorstores import InMemoryVectorStore

# Indexing - Loading documents
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Indexing - Splitting documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1  # Expecting exactly one document to be loaded
print(f"Total characters: {len(docs[0].page_content)}")
# print(docs[0].page_content[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Chunk size (characters)
    chunk_overlap=200,  # Chunk overlap (characters)
    add_start_index=True,  # Track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")
