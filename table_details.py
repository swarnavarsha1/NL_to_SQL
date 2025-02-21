import streamlit as st
from sqlalchemy import create_engine, inspect
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List
import os
import logging
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TableList(BaseModel):
    """List of relevant tables"""
    tables: List[str] = Field(description="List of table names that might be relevant")

def get_db_engine():
    try:
        connection_string = f"mysql+pymysql://{os.getenv('db_user')}:{os.getenv('db_password')}@{os.getenv('db_host')}/{os.getenv('db_name')}"
        engine = create_engine(connection_string)
        # Test connection
        with engine.connect() as conn:
            pass
        return engine
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def get_table_details() -> str:
    """Fetch all tables and their details from database"""
    try:
        engine = get_db_engine()
        if not engine:
            return ""
            
        inspector = inspect(engine)
        table_details = ""
        
        for table_name in inspector.get_table_names():
            table_details += f"Table: {table_name}\n"
            
            try:
                # Get columns
                columns = inspector.get_columns(table_name)
                table_details += "Columns:\n"
                for col in columns:
                    table_details += f"- {col['name']} ({col['type']})\n"
                
                # Get primary keys
                pks = inspector.get_pk_constraint(table_name)
                if pks['constrained_columns']:
                    table_details += f"Primary Keys: {', '.join(pks['constrained_columns'])}\n"
                
                # Get foreign keys
                fks = inspector.get_foreign_keys(table_name)
                if fks:
                    table_details += "Foreign Keys:\n"
                    for fk in fks:
                        table_details += f"- {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}\n"
                
                table_details += "\n"
            except Exception as e:
                logger.error(f"Error processing table {table_name}: {e}")
                table_details += f"Error processing table {table_name}: {str(e)}\n\n"
                
        return table_details
    except Exception as e:
        logger.error(f"Failed to fetch table details: {e}")
        return ""

def get_llm():
    """Initialize and return LLM with proper error handling"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        return ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            api_key=api_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return None

def table_chain(question: str) -> List[str]:
    """Get relevant tables for the given question"""
    table_details = get_table_details()
    table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
    The tables are:

    {table_details}

    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
    
    try:
        llm = get_llm()
        if not llm:
            # Fallback to returning all tables
            engine = get_db_engine()
            if engine:
                inspector = inspect(engine)
                return inspector.get_table_names()
            return []
            
        result = llm.with_structured_output(TableList).invoke(
            table_details_prompt + f"\n\nQuestion: {question}"
        )
        return result.tables
    except Exception as e:
        logger.error(f"Error in table chain: {e}")
        # Get all available tables as fallback
        try:
            engine = get_db_engine()
            if engine:
                inspector = inspect(engine)
                return inspector.get_table_names()
            return []
        except Exception as db_error:
            logger.error(f"Failed to get fallback tables: {db_error}")
            return []