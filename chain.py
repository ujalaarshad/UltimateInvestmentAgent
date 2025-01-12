# test4.py
import os
from datetime import datetime
import dotenv
from time import time
import streamlit as st
import logging
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    FireCrawlLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
# pip install docx2txt, pypdf
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 40
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# Function to stream the response of the LLM 
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# --- Indexing Phase ---

def load_doc_to_db():
    print("I  entered here")
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")
        print("exited 1")
        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")
            print("exited 2")


def load_url_to_db():
    """Load content from URL into the vector database with metadata."""
    url = st.session_state.get("rag_url", "").strip()
    if url and url not in st.session_state.rag_sources:
        try:
            api_key = os.getenv("FIRECRAWL_API_KEY")
            if not api_key:
                raise ValueError("FIRECRAWL_API_KEY environment variable not set")

            loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")
            docs = loader.load()

            # Add metadata to the loaded documents
            for doc in docs:
                doc.metadata.update(st.session_state.url_metadata)
                # Convert lists to strings in metadata
                doc.metadata = {k: ", ".join(map(str, v)) if isinstance(v, list) else v 
                              for k, v in doc.metadata.items()}

            st.session_state.rag_sources.append(url)
            st.session_state.documents_metadata[url] = st.session_state.url_metadata
            _split_and_load_docs(docs)
            st.toast(f"Content from URL loaded successfully: {url}", icon="✅")
            logger.info(f"Successfully loaded content from URL: {url}")

        except Exception as e:
            logger.error(f"Error loading content from URL {url}: {e}")
            st.error(f"Error loading content from URL: {e}")


# def initialize_vector_db(docs):
#     peristent_dir = os.path.join(os.getcwd(), "chroma_db")
    
#     os.makedirs(peristent_dir, exist_ok=True)
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_db = Chroma.from_documents(
#             documents=docs,
#             embedding=embedding,
#             collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
#             persist_directory= peristent_dir
#         )

#     # We need to manage the number of collections that we have in memory, we will keep the last 20
#     chroma_client = vector_db._client
#     collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
#     print("Number of collections:", len(collection_names))
#     while len(collection_names) > 20:
#         chroma_client.delete_collection(collection_names[0])
#         collection_names.pop(0)

#     return vector_db
# test4.py (modified version)

# def initialize_vector_db(docs):
#     peristent_dir = os.path.join(os.getcwd(), "chroma_db")
    
#     os.makedirs(peristent_dir, exist_ok=True)
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Include metadata from session state
#     documents_with_metadata = []
#     for doc in docs:
#         metadata = st.session_state.documents_metadata.get(doc.metadata.get('source', ''), {})
#         doc.metadata.update(metadata)
#         documents_with_metadata.append(doc)

#     vector_db = Chroma.from_documents(
#         documents=documents_with_metadata,
#         embedding=embedding,
#         collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
#         persist_directory=peristent_dir
#     )

#     chroma_client = vector_db._client
#     collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
#     while len(collection_names) > 20:
#         chroma_client.delete_collection(collection_names[0])
#         collection_names.pop(0)

#     return vector_db

def initialize_vector_db(docs):
    """Initialize vector database with metadata."""
    persistent_dir = os.path.join(os.getcwd(), "chroma_db")
    
    os.makedirs(persistent_dir, exist_ok=True)
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Include metadata from session state
    documents_with_metadata = []
    for doc in docs:
        source = doc.metadata.get('source', '')
        metadata = st.session_state.documents_metadata.get(
            source, 
            st.session_state.documents_metadata.get(doc.metadata.get('url', ''), {})
        )
        doc.metadata.update(metadata)
        documents_with_metadata.append(doc)

    vector_db = Chroma.from_documents(
        documents=documents_with_metadata,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
        persist_directory=persistent_dir
    )

    # Manage collections (keep last 20)
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db

# Rest of the code remains the same...
def load_url_to_db():
    """Load content from URL into the vector database with metadata."""
    metadata = st.session_state.get("url_metadata", {})
    url = metadata.get("url", "").strip()

    if not url:
        st.error("URL is missing from metadata. Please validate the metadata first.")
        return

    if url not in st.session_state.get("rag_sources", []):
        try:
            api_key = os.getenv("FIRECRAWL_API_KEY")
            if not api_key:
                raise ValueError("FIRECRAWL_API_KEY environment variable not set")

            loader = FireCrawlLoader(api_key=api_key, url=url, mode="scrape")
            docs = loader.load()

            # Add metadata to the loaded documents
            for doc in docs:
                doc.metadata.update(metadata)

                # Convert lists to strings in metadata
                doc.metadata = {k: ", ".join(map(str, v)) if isinstance(v, list) else v
                                for k, v in doc.metadata.items()}

            # Update session state with processed data
            st.session_state.rag_sources.append(url)
            st.session_state.documents_metadata[url] = metadata

            # Load documents into the database
            _split_and_load_docs(docs)
            st.toast(f"Content from URL loaded successfully: {url}", icon="✅")
            logger.info(f"Successfully loaded content from URL: {url}")

        except Exception as e:
            logger.error(f"Error loading content from URL {url}: {e}")
            st.error(f"Error loading content from URL: {e}")



def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
    )

    document_chunks = text_splitter.split_documents(docs)

    if ("vector_db" not in st.session_state) or st.session_state["vector_db"] is None:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


# --- Retrieval Augmented Generation (RAG) Phase ---
def handle_document_upload(uploaded_files):
    """Handle document uploads and metadata storage."""
    doc_category = st.selectbox("Category", ["Technical", "Business", "Legal", "Other"])
    doc_title = st.text_input("Title")
    doc_industry = st.selectbox("Industry", [
        "Technology", "Healthcare", "Finance", "Manufacturing", 
        "Retail", "Energy", "Entertainment", "Real Estate", "Other"
    ])
    doc_description = st.text_area("Description")

    if st.button("Upload"):
        for doc in uploaded_files:
            metadata = {
                "filename": doc.name,
                "title": doc_title,
                "category": doc_category,
                "industry": doc_industry,
                "description": doc_description,
                "uploaded_by": "awais",
                "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_type": doc.type,
                "source_type": "document"
            }
            st.session_state.documents_metadata[doc.name] = metadata
        load_doc_to_db()
        st.success("Documents uploaded successfully")
def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the industry, Focus more on industry in the conversation ,and must include in the query 'This focuses on [industry_name]', Dont include any information from yourself just generate query related to that industry. Always be specific and dont generate junk and if some industry information is not specific just add relevant keywords for that industry and if you cant find any industry just apologize, More focus on Industry name not risk and budget"),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    # print("Bro retrieve bhi hogayi")
    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """Act as a professional Investment Advisor AI. Engage with potential investors by asking these questions one at a time:

        Industry Interest: Any specific industry you're interested in? (e.g., technology, healthcare, etc.)
        Budget: What is your investment budget?
        Risk Tolerance: How do you describe your risk tolerance? (Low, Medium, High)
        Based on their responses, suggest tailored investment opportunities or ask follow-up questions to refine their preferences. When recommending companies, highlight their strengths, market insights, and reasons for investment. Address any concerns with data-driven arguments. If the user doesn't know their preferred industry, provide an overview of sectors like technology, healthcare, and entertainment.

        If no information is available about a sector, state so. Do not give unsolicited details or generalize about companies unless explicitly asked. Always use context for specific industry-related queries.
        NOTE: Always be specific and dont generate junk
        For further inquiries, contact us at helper@business.com.
        GENERATE INFORMATION IN SUCH A WAY THAT THEIR SHOULD BE COMPANY HEADING and its key features and why investor should invest in it , dont include any other unnecessary infomation, Dont tell very minute details about that company until user asks
        You should the context for user's queries to answer
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    # print("This is RAG response")
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})