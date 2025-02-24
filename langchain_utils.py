import os
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from db_utils import get_table_info, get_database
from prompts import sql_prompt, answer_prompt
import streamlit as st
import logging
import re

# Set logging to ERROR level to minimize output
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def clean_sql_query(sql: str) -> str:
    """Remove markdown formatting and clean SQL query"""
    sql = re.sub(r'```sql|```', '', sql)
    sql = sql.strip()
    return sql

def validate_database():
    """Validate database connection silently"""
    try:
        db = get_database()
        table_info = get_table_info(db)
        return db, table_info
    except Exception as e:
        logger.error(f"Database validation error: {e}")
        st.error("Unable to connect to the database. Please try again later.")
        return None, None

@st.cache_resource
def get_chain():
    try:
        db, table_info = validate_database()
        if not db or not table_info:
            raise Exception("Database validation failed")
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        execute_query = QuerySQLDataBaseTool(db=db)
        
        def generate_sql(inputs: dict) -> dict:
            try:
                question = inputs["question"].lower()
                prompt_value = sql_prompt.format(
                    question=question,
                    table_info=table_info
                )
                sql = llm.invoke(prompt_value).content.strip()
                sql = clean_sql_query(sql)
                return {"question": question, "query": sql}
            except Exception as e:
                logger.error(f"SQL generation error: {e}")
                raise
            
        def run_sql(inputs: dict) -> dict:
            try:
                query = inputs["query"]
                result = execute_query.invoke(query)
                return {
                    "question": inputs["question"],
                    "query": query,
                    "result": result if result else "No matching records found in the database."
                }
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                return {
                    "question": inputs["question"],
                    "query": query,
                    "result": f"Error executing query: {str(e)}"
                }
            
        def generate_answer(inputs: dict) -> str:
            try:
                if "No matching records found" in str(inputs["result"]):
                    return "I couldn't find any matching information in the database. Could you please try rephrasing your question?"
                    
                prompt_value = answer_prompt.format(**inputs)
                answer = llm.invoke(prompt_value).content.strip()
                return answer
            except Exception as e:
                logger.error(f"Answer generation error: {e}")
                return "I apologize, but I encountered an error while generating the answer."
        
        chain = (
            RunnableLambda(generate_sql) | 
            RunnableLambda(run_sql) | 
            RunnableLambda(generate_answer)
        )
        
        return chain
        
    except Exception as e:
        logger.error(f"Chain initialization error: {e}")
        st.error("Error setting up the system. Please try again later.")
        return None

def invoke_chain(question, messages):
    try:
        chain = get_chain()
        if not chain:
            return "System initialization failed. Please check the error messages above."
        
        response = chain.invoke({"question": question})
        return response
    except Exception as e:
        logger.error(f"Chain invocation error: {e}")
        return "I apologize, but I'm having trouble processing your question. Please try again."