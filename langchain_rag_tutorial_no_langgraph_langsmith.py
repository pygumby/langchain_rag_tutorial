"""LangChain RAG tutorial - no LangGraph, LangSmith"""

import getpass
import os

# Chat model
from langchain.chat_models import init_chat_model

# Embeddings model
from langchain_openai import OpenAIEmbeddings

# Indexing - Loading documents
from langchain_community.document_loaders import WebBaseLoader
import bs4

# Indexing - Splitting documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Indexing - Storing documents
from langchain_core.vectorstores import InMemoryVectorStore

# Retrieval and generation
from langchain_core.prompts import PromptTemplate


os.environ["LANGSMITH_TRACING"] = "true"

if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith: ")

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

# Embed and store chunks
document_ids = vector_store.add_documents(documents=all_splits)
# print(document_ids[:3])

# Retrieve and generate
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "Thanks you for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate.from_template(template)

# example_message = prompt.invoke(
#     {
#         "context": "(context goes here)",
#         "question": "(question goes here)",
#     }
# ).to_messages()

# assert len(example_message) == 1
# print(example_message[0].content)

question = "What is task decomposition?"

retrieved_docs = vector_store.similarity_search(question)
docs_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
prompt = prompt.invoke({"question": question, "context": docs_content})

answer = llm.invoke(prompt)

print(answer.content)
