"""LangChain RAG tutorial, part 2, agents"""

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

# Retrieval and generation, chains
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


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
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query"""

    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


memory = MemorySaver()

# Leverage agentic capabilities using LangGraph's pre-built ReAct agent constructor
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

# Test the app
input_message = (
    "What is the standard method for task decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    event["messages"][-1].pretty_print()
