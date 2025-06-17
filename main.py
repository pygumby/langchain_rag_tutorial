import getpass
import os

# Chat model
from langchain.chat_models import init_chat_model

# Embeddings model
from langchain_openai import OpenAIEmbeddings

# Vector store
from langchain_core.vectorstores import InMemoryVectorStore

# Indexing – Loading documents
from langchain_community.document_loaders import WebBaseLoader
import bs4

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
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
print(docs[0].page_content[:500])
