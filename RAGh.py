import logging
import time
import os
import faiss  # FAISS dependency for vector search
from langchain import hub
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langsmith import Client
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain_community.embeddings import OllamaEmbeddings


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the language for Chainlit to English
os.environ['LANGUAGE'] = 'en-US'

# Initialize LangSmith for debugging
langsmith_client = Client(api_key="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")

# Set up QA Chain prompt
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")

# Load the LLM (Ollama)
def load_llm():
    llm = OllamaLLM(
        model="mistral-small",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

DATA_PATH = "data/"
CHROMADB_PATH = "vectorstores/db/chroma/"
FAISSDB_PATH = "vectorstores/db/faiss/"
model_config = {}
model_config['protected_namespaces'] = ()

# Initialize the vectorstore database (optional step)
def initialize_vectorstore(db_type="faiss"):
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    logger.info(f"Processed {len(documents)} pdf files")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150)
    texts = text_splitter.split_documents(documents)
    
    embedding_function = GPT4AllEmbeddings()

    if db_type == "faiss":
        embedding_dimension = 384  # Set this to the correct dimension for your embedding model
        vectorstore = FAISS.from_documents(documents=texts, embedding=embedding_function)
        vectorstore.save_local(FAISSDB_PATH)
    else:
        vectorstore = Chroma.from_documents(documents=texts, embedding=embedding_function, persist_directory=CHROMADB_PATH)
        vectorstore.persist()
    
    return vectorstore

# Retrieval QA Chain setup with timing and memory
def retrieval_qa_chain(llm, vectorstore):
    class TimedCallbackHandler(StreamingStdOutCallbackHandler):
        def __init__(self):
            super().__init__()
            self.retrieval_time = 0
            self.model_time = 0

        async def on_retriever_start(self):
            self.retrieval_start_time = time.time()

        async def on_retriever_end(self):
            self.retrieval_time = time.time() - self.retrieval_start_time
            logger.info(f"Vectorstore retrieval took {self.retrieval_time:.2f} seconds")

        async def on_llm_start(self):
            self.model_start_time = time.time()

        async def on_llm_end(self):
            self.model_time = time.time() - self.model_start_time
            logger.info(f"Model response took {self.model_time:.2f} seconds")

        def log_times(self):
            logger.info(f"Final log - Vectorstore retrieval time: {self.retrieval_time:.2f} seconds")
            logger.info(f"Final log - Model response time: {self.model_time:.2f} seconds")

    timed_callback_handler = TimedCallbackHandler()

    # Add a memory component for tracking conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        memory=memory,
        return_source_documents=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler(), timed_callback_handler]),
        output_key="result"  # Specify 'result' as the output key to store in memory
    )

    timed_callback_handler.log_times()

    return qa_chain

# QA Bot logic to load LLM and initialize QA chain
def qa_bot(initialize_db=False, db_type="faiss"):
    llm = load_llm()

    if initialize_db:
        vectorstore = initialize_vectorstore(db_type=db_type)
    else:
        if db_type == "faiss":
            vectorstore = FAISS.load_local(FAISSDB_PATH, embedding_function=GPT4AllEmbeddings())
        else:
            vectorstore = Chroma(persist_directory=CHROMADB_PATH, embedding_function=GPT4AllEmbeddings())

    qa = retrieval_qa_chain(llm, vectorstore)
    return qa

# Chainlit startup
@cl.on_chat_start
async def start():
    msg = cl.Message(content="Willkommen beim Research Info Bot! Möchten Sie die Vectorstore-Datenbank initialisieren?")
    await msg.send()

    options_msg = cl.Message(
        content="Antworten Sie mit 'yes', um die Datenbank zu initialisieren, oder 'no', um fortzufahren, ohne sie zu initialisieren."
    )
    await options_msg.send()

# Handle all user responses
@cl.on_message
async def handle_message(message):
    initialization_done = cl.user_session.get("initialization_done", False)

    if not initialization_done:
        if message.content.lower() == "yes":
            chain = qa_bot(initialize_db=True, db_type="faiss")
            msg = cl.Message(content="Die Vectorstore-Datenbank wird initialisiert. Bitte warten Sie...")
            await msg.send()
        elif message.content.lower() == "no":
            chain = qa_bot(initialize_db=False, db_type="faiss")
            msg = cl.Message(content="Fortfahren, ohne die Vectorstore-Datenbank zu initialisieren.")
            await msg.send()
        else:
            msg = cl.Message(content="Ungültige Antwort. Bitte antworten Sie mit 'yes' oder 'no'.")
            await msg.send()
            return

        cl.user_session.set("chain", chain)
        cl.user_session.set("initialization_done", True)
        query_msg = cl.Message(content="Was ist Ihre Anfrage?")
        await query_msg.send()

    else:
        chain = cl.user_session.get("chain")
        if chain is None:
            await cl.Message(content="Es wurde kein Verarbeitungskette (chain) gefunden. Bitte starten Sie die Konversation erneut.").send()
            return

        start_time = time.time()

        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )

        res = await chain.ainvoke(message.content, callbacks=[cb])

        total_time = time.time() - start_time
        logger.info(f"Total query processing time: {total_time:.2f} seconds")

        answer = res["result"].replace(".", ".\n")
        sources = res["source_documents"]

        if sources:
            detailed_sources = []
            for doc in sources:
                source_info = (
                    f"- **File Name**: {doc.metadata.get('source', 'Unknown')}\n"
                    f"  **Document ID**: {doc.metadata.get('document_id', 'N/A')}\n"
                    f"  **Page Number**: {doc.metadata.get('page', 'N/A')}\n"
                    f"  **Other Metadata**: {doc.metadata}\n"
                )
                detailed_sources.append(source_info)
            answer += "\n### Sources:\n" + "\n".join(detailed_sources)
        else:
            answer += "\nNo sources found."

        await cl.Message(content=answer).send()