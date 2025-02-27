import streamlit as st
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from db_utils import get_database, get_table_info
from operator import itemgetter
from dotenv import load_dotenv
import os
import re

# Load environment variables at module level
load_dotenv()

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")
    confidence: float = Field(description="Confidence score (0-1) that this table is relevant to the question.")

def get_tables(tables: List[Table]) -> List[str]:
    """Convert Table objects to list of table names, filtering by confidence threshold."""
    # Sort by confidence score and filter tables with confidence > 0.3
    sorted_tables = sorted(tables, key=lambda x: x.confidence, reverse=True)
    return [table.name for table in sorted_tables if table.confidence > 0.3]

def normalize_question(question: str) -> str:
    """Normalize the question to improve matching."""
    # Convert to lowercase
    question = question.lower()
    # Replace hyphens with spaces for consistent matching
    question = question.replace('-', ' ')
    # Replace multiple spaces with a single space
    question = re.sub(r'\s+', ' ', question)
    
    # Add recommendation keyword mapping
    recommendation_keywords = ["good", "best", "recommend", "popular", "favorite", "suggestion"]
    if any(keyword in question for keyword in recommendation_keywords):
        question += " recommendation popular"
    
    # Add category keywords for common food-related queries
    if "vegetarian" in question or "veg " in question:
        question += " vegetarian dietary"
        
    if "non vegetarian" in question or "non veg" in question:
        question += " non-vegetarian meat"
    
    return question.strip()

@st.cache_resource
def get_table_selection_chain():
    """Create the table selection chain with improved system message."""
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        db = get_database()
        table_info = get_table_info(db)
        
        system_message = f"""Analyze the user's question and determine which database tables are most relevant to answering it.
The database schema is:

{table_info}

For each table, provide:
1. The table name
2. A confidence score (0-1) indicating how relevant this table is to the question

Consider:
- The subject matter of the question (what entity or information is being asked about)
- The type of attributes being queried
- Similar questions might need different tables depending on specific details

Return ALL tables that might contain relevant information, with appropriate confidence scores.
"""
        
        return create_extraction_chain_pydantic(Table, llm, system_message=system_message)
    except Exception as e:
        st.error(f"Error creating table selection chain: {str(e)}")
        raise

def select_relevant_tables(question: str) -> List[str]:
    """Select relevant tables for a given question with improved preprocessing."""
    try:
        chain = get_table_selection_chain()
        normalized_question = normalize_question(question)
        
        # Execute chain with normalized question
        tables_with_confidence = chain.run(normalized_question)
        
        # Get table names from results, sorted by confidence
        table_names = get_tables(tables_with_confidence)
        
        # If no tables meet the confidence threshold, return all tables as fallback
        if not table_names:
            db = get_database()
            engine = db._engine
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            
        return table_names
    except Exception as e:
        st.error(f"Error selecting relevant tables: {str(e)}")
        # Return all tables as fallback in case of error
        db = get_database()
        engine = db._engine
        from sqlalchemy import inspect
        inspector = inspect(engine)
        return inspector.get_table_names()