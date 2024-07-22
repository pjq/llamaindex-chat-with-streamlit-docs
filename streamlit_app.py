import streamlit as st
import openai
import os
import logging
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PERSIST_DIR = "./storage"
DATA_DIR = "./mdc-docs/"

st.set_page_config(page_title="Chat with the MDC/PPX docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
openai.base_url= st.secrets.base_url
st.title("Chat with the MDC/PPX docs, powered by LlamaIndex 💬🦙")
st.info("Check out the full MDC tutorial [blog post](https://pages.github.tools.sap/sfmobile/mdc-docs/)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about SuccessFactors MDC document!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    def create_index(data_dir, persist_dir):
        logging.info(f"Creating index from documents in {data_dir}")
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
            logging.info(f"Created directory at {persist_dir}")

        index.storage_context.persist(persist_dir=persist_dir)
        logging.info(f"Index created and persisted to {persist_dir}")
        return index

    def load_or_create_index(persist_dir, data_dir):
        if not os.path.exists(persist_dir):
            logging.info(f"No existing storage found at {persist_dir}. Creating new index.")
            return create_index(data_dir, persist_dir)
        else:
            logging.info(f"Loading existing index from {persist_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            return load_index_from_storage(storage_context)

    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=0.2,
        system_prompt="""You are an expert on 
        the SAP SuccessFactors Mobile MDC(Metadata driven control) and your 
        job is to answer technical questions. 
        Assume that all questions are related 
        to the MDC. Keep 
        your answers technical and based on 
        facts – do not hallucinate features.""",
    )
    return load_or_create_index(PERSIST_DIR, DATA_DIR)

index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)