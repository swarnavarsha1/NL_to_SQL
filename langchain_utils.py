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
from typing import Dict, List, Any
from sqlalchemy import inspect, text

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

def perform_fallback_query(db, question: str) -> Dict[str, Any]:
    """
    Perform fallback text search across all tables to find potential matches
    when primary query returns no results.
    """
    try:
        engine = db._engine
        inspector = inspect(engine)
        fallback_results = {}
        
        normalized_question = question.lower().replace('-', ' ').strip()
        search_terms = normalized_question.split()
        
        # Only use meaningful search terms (filter out common words)
        stop_words = {'what', 'where', 'when', 'how', 'who', 'list', 'show', 'tell', 'give', 'are', 'is', 'the', 'a', 'an', 'can', 'you', 'your', 'me', 'i', 'my', 'available'}
        search_terms = [term for term in search_terms if term not in stop_words and len(term) > 2]
        
        # If we have valid search terms
        if search_terms:
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                text_columns = [col['name'] for col in columns if 'varchar' in str(col['type']).lower() or 'text' in str(col['type']).lower()]
                
                if text_columns:
                    # For each text column, create a LIKE condition for each search term
                    for column in text_columns:
                        for term in search_terms:
                            query = f"SELECT * FROM {table_name} WHERE LOWER({column}) LIKE :search_term LIMIT 5"
                            try:
                                with engine.connect() as conn:
                                    result = conn.execute(text(query), {"search_term": f"%{term}%"})
                                    results = [dict(row) for row in result]
                                    if results:
                                        if table_name not in fallback_results:
                                            fallback_results[table_name] = []
                                        fallback_results[table_name].extend(results)
                            except Exception as e:
                                logger.error(f"Error in fallback query for {table_name}.{column}: {e}")
        
        return fallback_results
    except Exception as e:
        logger.error(f"Error in fallback search: {e}")
        return {}

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
                question = inputs["question"]
                
                # Special handling for FAQ-type questions by using a more generic approach
                normalized_question = question.lower()
                
                # Look for common FAQ patterns without hardcoding table names or specific content
                is_faq_pattern = any(pattern in normalized_question for pattern in 
                                    ["best", "popular", "recommend", "special", "favorite", 
                                     "signature", "house", "famous"])
                
                if is_faq_pattern:
                    # Try to detect which tables might have FAQ-style content
                    db = get_database()
                    engine = db._engine
                    inspector = inspect(engine)
                    
                    for table_name in inspector.get_table_names():
                        columns = inspector.get_columns(table_name)
                        col_names = [col["name"].lower() for col in columns]
                        
                        # Look for tables that have FAQ-like column pairs (question/answer, etc.)
                        has_question_col = any("question" in col.lower() for col in col_names)
                        has_answer_col = any("answer" in col.lower() for col in col_names)
                        
                        if has_question_col and has_answer_col:
                            # Get the actual column names (not the lowercase versions)
                            question_col = next(col["name"] for col in columns if "question" in col["name"].lower())
                            answer_col = next(col["name"] for col in columns if "answer" in col["name"].lower())
                            
                            # Create a direct query to the FAQ-like table
                            keywords = re.findall(r'\b\w+\b', normalized_question)
                            search_terms = [term for term in keywords if len(term) > 3 and term not in 
                                           ["what", "where", "when", "which", "your", "best", "popular", "have", "tell"]]
                            
                            if search_terms:
                                conditions = []
                                for term in search_terms:
                                    conditions.append(f"{question_col} LIKE '%{term}%'")
                                
                                direct_query = f"SELECT {answer_col} FROM {table_name} WHERE {' OR '.join(conditions)}"
                                return {"question": question, "query": direct_query}
                
                # Default to LLM-generated query if no special case matched
                prompt_value = sql_prompt.format(
                    question=question,
                    table_info=table_info
                )
                sql = llm.invoke(prompt_value).content.strip()
                sql = clean_sql_query(sql)
                return {"question": question, "query": sql}
            except Exception as e:
                logger.error(f"SQL generation error: {e}")
                # Fallback to standard query generation on exception
                prompt_value = sql_prompt.format(
                    question=question,
                    table_info=table_info
                )
                sql = llm.invoke(prompt_value).content.strip()
                sql = clean_sql_query(sql)
                return {"question": question, "query": sql}
            
        def run_sql(inputs: dict) -> dict:
            try:
                query = inputs["query"]
                result = execute_query.invoke(query)
                
                # Check if we got an empty result
                if not result or "No matching records" in result:
                    # Try fallback search if main query returned no results
                    fallback_results = perform_fallback_query(db, inputs["question"])
                    
                    # If fallback found something
                    if fallback_results:
                        return {
                            "question": inputs["question"],
                            "query": query,
                            "result": result,
                            "fallback_results": fallback_results
                        }
                
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
                # Check if we have fallback results that might be useful
                if "fallback_results" in inputs and inputs["fallback_results"]:
                    # Add the fallback results to the prompt
                    fallback_info = "While we couldn't find exact matches, here are some potentially related items:\n"
                    for table, rows in inputs["fallback_results"].items():
                        fallback_info += f"\nFrom {table}:\n"
                        for row in rows[:3]:  # Limit to 3 results per table
                            fallback_info += f"- {row}\n"
                    
                    # Add fallback info to the result
                    if "No matching records" in str(inputs["result"]):
                        inputs["result"] = fallback_info
                    else:
                        inputs["result"] += "\n\nAdditional potential matches:\n" + fallback_info
                
                if "No matching records found" in str(inputs["result"]) and not inputs.get("fallback_results"):
                    # Suggest alternative phrasings based on the question
                    question = inputs["question"].lower()
                    suggestions = []
                    
                    if "time" in question or "hour" in question:
                        suggestions.append("Try asking about specific meal times like 'What are your lunch hours?' or 'When is dinner served?'")
                    
                    if "menu" in question or "dish" in question or "food" in question or "item" in question:
                        suggestions.append("Try asking about specific categories like 'What vegetarian dishes do you have?' or 'Do you have spicy items?'")
                    
                    if not suggestions:
                        suggestions = ["Could you try rephrasing your question with more specific details?"]
                    
                    suggestion_text = " ".join(suggestions)
                    return f"I couldn't find any matching information in the database. {suggestion_text}"
                    
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