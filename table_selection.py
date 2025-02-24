import streamlit as st
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic import BaseModel, Field  # Updated import
from langchain_openai import ChatOpenAI
from typing import List
from db_utils import get_database, get_table_info
from operator import itemgetter
from dotenv import load_dotenv
import os

# Load environment variables at module level
load_dotenv()

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

def get_tables(tables: List[Table]) -> List[str]:
    """Convert Table objects to list of table names."""
    return [table.name for table in tables]

@st.cache_resource
def get_table_selection_chain():
    """Create the table selection chain."""
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        db = get_database()
        table_info = get_table_info(db)
        
        system_message = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question.
The database schema is:

{table_info}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
        
        return create_extraction_chain_pydantic(Table, llm, system_message=system_message)
    except Exception as e:
        st.error(f"Error creating table selection chain: {str(e)}")
        raise

def select_relevant_tables(question: str) -> List[str]:
    """Select relevant tables for a given question."""
    try:
        chain = get_table_selection_chain()
        return {"input": itemgetter("question")} | chain | get_tables
    except Exception as e:
        st.error(f"Error selecting relevant tables: {str(e)}")
        raise