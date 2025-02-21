import os
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from table_details import get_table_details, get_db_engine, get_llm
from prompts import sql_prompt, answer_prompt
import streamlit as st
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment variables")

# Database configuration
db_config = {
    "user": os.getenv("db_user"),
    "password": os.getenv("db_password"),
    "host": os.getenv("db_host"),
    "database": os.getenv("db_name")
}

@st.cache_resource
def get_chain():
    try:
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        # Create database connection string
        db_uri = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
        db = SQLDatabase.from_uri(db_uri)
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o",  # Use the appropriate model name
            temperature=0,
            api_key=api_key
        )
        
        # Create SQL execution tool
        execute_query = QuerySQLDataBaseTool(db=db)
        
        # Create the chain. Notice that we explicitly extract .content.strip()
        chain = (
            RunnablePassthrough.assign(
                query=lambda x: llm.invoke(
                    sql_prompt.format(
                        question=x["question"],
                        table_info=x["table_info"]
                    )
                ).content.strip(),
                result=lambda x: execute_query.invoke({"query": x["query"]})
            ) | answer_prompt | llm | StrOutputParser()
        )
        
        logger.debug("Chain constructed successfully")
        return chain
        
    except Exception as e:
        logger.error(f"Error initializing chain: {e}")
        st.error(f"Error initializing chain: {str(e)}")
        return None

def invoke_chain(question, messages):
    try:
        chain = get_chain()
        if not chain:
            return "Sorry, I'm having trouble connecting to the language model. Please check your OpenAI API key configuration."
        
        # Generate the SQL query first by using get_llm and extracting the content
        table_info = get_table_details()
        llm_instance = get_llm()
        sql_response = llm_instance.invoke(
            sql_prompt.format(question=question, table_info=table_info)
        )
        # Ensure we extract the text content
        query = sql_response.content.strip()
        
        chain_input = {
            "question": question,
            "table_info": table_info,
            "query": query  # Ensure the query key exists
        }
        
        logger.debug(f"Chain input: {chain_input}")
        
        response = chain.invoke(chain_input)
        logger.debug(f"Chain response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error invoking chain: {e}")
        return f"An error occurred: {str(e)}"
