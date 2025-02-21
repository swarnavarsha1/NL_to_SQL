import streamlit as st

st.set_page_config(
    page_title="Database Query Assistant",
    layout="wide"
)

from openai import OpenAI
from langchain_utils import invoke_chain
from dotenv import load_dotenv
import os
import logging
import traceback
from table_details import get_table_details, get_db_engine, get_llm

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add caching to the imported functions
get_db_engine = st.cache_resource(get_db_engine)
get_table_details = st.cache_data(get_table_details)
get_llm = st.cache_resource(get_llm)

def init_app():
    try:
        # Load environment variables
        load_dotenv()
        logger.debug("Environment variables loaded")
        
        # Initialize environment checks
        env_checks = {
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "db_user": bool(os.getenv("db_user")),
            "db_password": bool(os.getenv("db_password")),
            "db_host": os.getenv("db_host"),
            "db_name": os.getenv("db_name")
        }
        
        # Get table details
        table_info = get_table_details()
        logger.debug(f"Retrieved table details: {bool(table_info)}")
        
        return env_checks, table_info
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Application initialization failed: {str(e)}")
        return None, None

# Initialize app
env_checks, table_info = init_app()

# Debug information
with st.expander("Debug Information", expanded=True):
    st.write("Environment Check:")
    if env_checks:
        for var, value in env_checks.items():
            if var in ["db_host", "db_name"]:
                st.write(f"- {var}: {value}")
            else:
                st.write(f"- {var}: {'✅ Set' if value else '❌ Missing'}")
    
    st.write("\nDatabase Tables:")
    if table_info:
        st.code(table_info)
    else:
        st.error("No table information available")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.debug("Initialized messages in session state")

# App title and description
st.title("🔍 Database Query Assistant")
st.markdown("""
This app helps you query your database using natural language. 
Just ask a question about your data, and I'll help you find the answer!
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your database..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = invoke_chain(prompt, st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                logger.error(traceback.format_exc())
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check the debug information above for more details.")