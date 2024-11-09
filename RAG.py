import logging
import time
import os
import faiss  # FAISS dependency for vector search
from langchain import hub
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage  # Import for message type checking
from langsmith import Client
import chainlit as cl
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain_community.embeddings import OllamaEmbeddings

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# Set the language for Chainlit to English
os.environ['LANGUAGE'] = 'en-US'

# Initialize LangSmith for debugging (Optional)
# langsmith_client = Client(api_key="YOUR_API_KEY_HERE")

# Set up QA Chain prompt
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-mistral")

# Define a custom prompt template with your specific text
CUSTOM_PROMPT_TEMPLATE = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are an advanced research assistant. Consider the previous conversation context:\n{context}\n"
            "Respond with detailed insights based on the question and available context from retrieved documents:\n{context}"
        ),
        HumanMessagePromptTemplate.from_template("{question}")
    ]  ,
    input_variables=["chat_history", "context"]  # Define input variables here
)


DATA_PATH = "data/"
CHROMADB_PATH = "vectorstores/db/chroma/"
FAISSDB_PATH = "vectorstores/db/faiss/"
model_config = {'protected_namespaces': ()}
DB_TYPE = "faiss"

# Initialize ConversationBufferMemory
message_history = ChatMessageHistory()
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="result",
        chat_memory=message_history,
        return_messages=True,
    )

# Load the LLM (Ollama)
def load_llm():
    try:
        llm = OllamaLLM(
            model="mistral-small",
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        raise

# Vectorstore Initialization with Error Handling
def initialize_vectorstore(db_type="faiss"):
    try:
        loader = PyPDFDirectoryLoader(DATA_PATH)
        documents = loader.load()
        logger.info(f"Processed {len(documents)} PDF files.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=400)
        texts = text_splitter.split_documents(documents)
        
        embedding_function = GPT4AllEmbeddings()
        
        if db_type == "faiss":
            embedding_dimension = 384  # Set this to the correct dimension for your embedding model
            vectorstore = FAISS.from_documents(documents=texts, embedding=embedding_function)
            vectorstore.save_local(FAISSDB_PATH)
        else:
            vectorstore = Chroma.from_documents(documents=texts, embedding=embedding_function, persist_directory=CHROMADB_PATH)
            #vectorstore.persist()
        
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize vectorstore: {e}")
        raise

# QA Bot logic to load LLM and initialize QA chain with error handling
def qa_bot(initialize_db=False, db_type=DB_TYPE):
    try:
        llm = load_llm()

        if initialize_db:
            vectorstore = initialize_vectorstore(db_type=db_type)
        else:
            if db_type == "faiss":
                vectorstore = FAISS.load_local(FAISSDB_PATH, embeddings=GPT4AllEmbeddings(),allow_dangerous_deserialization=True)
            else:
                vectorstore = Chroma(persist_directory=CHROMADB_PATH, embedding_function=GPT4AllEmbeddings(),allow_dangerous_deserialization=True)

        qa = retrieval_qa_chainC(llm, vectorstore)
        return qa
    except Exception as e:
        logger.error(f"Failed to initialize QA Bot: {e}")
        raise

# Chainlit startup message
@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the Research Info Bot! Would you like to initialize the vectorstore database?").send()
    await cl.Message(content="Reply with 'yes' to initialize or 'no' to proceed without initialization.").send()
    print(f"Input Variables QA_CHAIN_PROMPT: {QA_CHAIN_PROMPT.input_variables}")
    print(f"Input Variables CUSTOM_PROMPT: {CUSTOM_PROMPT_TEMPLATE.input_variables}")



# Then, use this template within the RetrievalQA chain setup
def retrieval_qa_chainC(llm, vectorstore):
    class TimedCallbackHandler(StreamingStdOutCallbackHandler):
        def __init__(self):
            super().__init__()
            self.retrieval_time = 0
            self.model_time = 0

        async def on_retriever_start(self):
            self.retrieval_start_time = time.time()
            logger.debug("Retriever activation started.")

        async def on_retriever_end(self):
            self.retrieval_time = time.time() - self.retrieval_start_time
            logger.debug(f"Retriever activation ended. Duration: {self.retrieval_time:.2f} seconds")

        async def on_llm_start(self):
            self.model_start_time = time.time()
            logger.debug("LLM processing started.")

        async def on_llm_end(self):
            self.model_time = time.time() - self.model_start_time
            logger.debug(f"LLM processing ended. Duration: {self.model_time:.2f} seconds")

    logger.debug("Setting up custom QA chain with user-defined prompt template.")
    
    # Use RetrievalQA with context-aware prompt template    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
        chain_type_kwargs={
            "prompt": CUSTOM_PROMPT_TEMPLATE,
            "document_variable_name": "context"  # Specify that the retrieved documents should be passed as 'context'
        },
        memory=memory,
        return_source_documents=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler(), TimedCallbackHandler()]),
        verbose=True
    )
    return qa_chain

# Handle incoming messages and display retrieved documents
@cl.on_message
async def handle_message(message):
    initialization_done = cl.user_session.get("initialization_done", False)

    if not initialization_done:
        if message.content.lower() == "yes":
            await cl.Message(content="Initializing the vectorstore database. Please wait...").send()
            chain = qa_bot(initialize_db=True, db_type=DB_TYPE)
        elif message.content.lower() == "no":
            await cl.Message(content="Proceeding without initializing the vectorstore database.").send()
            chain = qa_bot(initialize_db=False, db_type=DB_TYPE)
        else:
            await cl.Message(content="Invalid response. Please reply with 'yes' or 'no'.").send()
            return

        cl.user_session.set("chain", chain)
        cl.user_session.set("initialization_done", True)
        await cl.Message(content="What is your query?").send()

    else:
        
        qa_chain = cl.user_session.get("chain")
        if qa_chain is None:
            await cl.Message(content="No processing chain found. Please restart the conversation.").send()
            return

        start_time = time.time()

        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        #print(f"<<<{message.content}/{chat_history}/{chat_history_content}>>>>>\n")

        try:
            # Invoke the chain and retrieve the response
            # Check if chat_history_content is already defined
            # Retrieve chat history from memory and format it
            chat_history = memory.chat_memory.messages
            chat_history_content = "\n".join(
                [f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}" for msg in chat_history]
            ) if chat_history else "HugoPortisch"  # Initialize to empty if chat_history is not present

            logger.debug(f"chat_history:({chat_history_content})")
            # print(QA_CHAIN_PROMPT.format(context="Background information on the topic", question="What is climate change?"))
            #res = await chain.ainvoke({"query":message.content,"chat_history": chat_history_content}, callbacks=[cb])
            res = await qa_chain.ainvoke({
                        "query": message.content,
                        "chat_history": chat_history_content
                    }, callbacks=[cb])
            print(res.keys())
            
            total_time = time.time() - start_time
            logger.info(f"Total query processing time: {total_time:.2f} seconds")

            # Extract the result from the response without using `output_key`
            answer = res.get("result", "").replace(".", ".\n")  # Extracts result text

            sources = res.get("source_documents", [])
            prevdoc = None
            
            # Display the retrieved sources
            count=0
            if sources:
                detailed_sources = []
                for doc in sources:
                    count +=1
                    logger.info(f">>>>>{count}-{doc.metadata.get('page','N/A')}\n")
                    if prevdoc == doc:
                        break
                    
                    source_info = (
                        f"- **File Name**: {doc.metadata.get('source', 'Unknown')}  #{doc.metadata.get('page', 'N/A')}\n"
                        #f"  **Document ID**: {doc.metadata.get('document_id', 'N/A')}\n"
                        f"  **Snippet**: {doc.page_content[:400]}...\n"  # Show a snippet of the content
                    )
                    detailed_sources.append(source_info)
                    prevdoc=doc
                answer += f"{count}-"
                answer += "\n### Sources Found in Vector Database:\n" + "\n".join(detailed_sources)
            else:
                answer += "\nNo sources found in vector database."

            # Retrieve and log the chat history to console
            chat_history = memory.chat_memory.messages
            chat_history_content = "\n".join(
                [f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Bot: {msg.content}" for msg in chat_history]
            )
            logger.info("\nChat History:\n" + chat_history_content)
            
            # answer += "\n\n### Chat History:\n" + chat_history_content
            await cl.Message(content=answer).send()

        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            await cl.Message(content="An error occurred while processing your query. Please try again.").send()