"""LangChain RAG tutorial, part 1"""

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
from langchain import hub
from langchain_core.documents import Document
from typing import List, TypedDict
from langgraph.graph import START, StateGraph

# Query analysis
from typing import Literal, Annotated


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

# Add metadata
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    if i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# Embed and store chunks
document_ids = vector_store.add_documents(documents=all_splits)
# print(document_ids[:3])

# Retrieve and generate
prompt = hub.pull("rlm/rag-prompt")

# example_message = prompt.invoke(
#     {
#         "context": "(context goes here)",
#         "question": "(question goes here)",
#     }
# ).to_messages()

# assert len(example_message) == 1
# print(example_message[0].content)


class Search(TypedDict):
    """Search query"""

    query: Annotated[str, ..., "Search query to run"]
    section: Annotated[Literal["beginning", "middle", "end"], ..., "Section to query"]


class State(TypedDict):
    """State of the RAG system"""

    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    """Analyze query step"""

    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    """Retrieval step"""

    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query=query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    """Generation step"""

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {
            "question": state["question"],
            "context": docs_content,
        }
    )
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

for step in graph.stream(
    {"question": "What does the end of the post say about task decomposition?"},
    stream_mode="updates",
):
    print(f"{step}")
