from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, inspect, text
import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

def get_all_table_names(engine) -> List[str]:
    """Get all table names from the database."""
    inspector = inspect(engine)
    return inspector.get_table_names()

def get_database() -> SQLDatabase:
    """Create and return a SQLDatabase instance using environment variables."""
    db_user = os.getenv("db_user")
    db_password = os.getenv("db_password")
    db_host = os.getenv("db_host")
    db_name = os.getenv("db_name")
    
    uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    
    # First create engine to get table names
    engine = create_engine(uri)
    tables = get_all_table_names(engine)
    
    # Create SQLDatabase with all tables included
    return SQLDatabase.from_uri(uri, include_tables=tables)

def analyze_schema(db: SQLDatabase) -> dict:
    """Analyze database schema and extract metadata about tables and their purposes."""
    engine = db._engine
    inspector = inspect(engine)
    schema_info = {}
    
    for table_name in inspector.get_table_names():
        # Get table structure
        columns = inspector.get_columns(table_name)
        sample_data = []
        
        # Get sample data to understand content
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 5"))
                sample_data = [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error getting sample data for {table_name}: {e}")
        
        # Determine table purpose based on column analysis
        purpose = infer_table_purpose(table_name, columns, sample_data)
        
        schema_info[table_name] = {
            "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns],
            "sample_data": sample_data,
            "relationships": inspector.get_foreign_keys(table_name),
            "purpose": purpose
        }
    
    return schema_info

def infer_table_purpose(table_name: str, columns: List[Dict], sample_data: List[Dict]) -> str:
    """
    Infer the purpose of a table based on its name, columns, and sample data.
    This helps the LLM understand what kind of information each table contains.
    """
    # Extract column names for easier analysis
    column_names = [col["name"].lower() for col in columns]
    
    # Check for common patterns in column names
    has_product_columns = any(col in column_names for col in ["product", "item", "name", "description", "price", "category"])
    has_user_columns = any(col in column_names for col in ["user", "customer", "email", "phone", "address"])
    has_transaction_columns = any(col in column_names for col in ["order", "transaction", "payment", "date", "time"])
    has_location_columns = any(col in column_names for col in ["location", "address", "city", "state", "country"])
    has_time_columns = any(col in column_names for col in ["hour", "time", "schedule", "availability"])
    has_question_columns = any(col in column_names for col in ["question", "answer", "faq"])
    
    # Check table name for clues
    table_name_lower = table_name.lower()
    
    if "menu" in table_name_lower or "item" in table_name_lower or has_product_columns:
        return "Contains information about menu items, products, or offerings"
    elif "user" in table_name_lower or "customer" in table_name_lower or has_user_columns:
        return "Contains information about users or customers"
    elif "order" in table_name_lower or "transaction" in table_name_lower or has_transaction_columns:
        return "Contains information about orders or transactions"
    elif "location" in table_name_lower or has_location_columns:
        return "Contains information about locations or addresses"
    elif "hour" in table_name_lower or "schedule" in table_name_lower or has_time_columns:
        return "Contains information about hours, schedules, or availability"
    elif "faq" in table_name_lower or has_question_columns:
        return "Contains frequently asked questions and answers"
    elif "hotel" in table_name_lower or "restaurant" in table_name_lower:
        return "Contains information about the establishment, services, or policies"
    else:
        # Generic fallback that uses the table name
        return f"Contains information related to {table_name.replace('_', ' ')}"

def get_table_info(db: SQLDatabase) -> str:
    """Get detailed information about all tables in the database."""
    engine = db._engine
    inspector = inspect(engine)
    table_info = []
    
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        fks = inspector.get_foreign_keys(table_name)
        pks = inspector.get_pk_constraint(table_name)
        
        # Analyze sample data to improve table description
        sample_data = []
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3"))
                sample_data = []
                for row in result:
                    try:
                        sample_data.append({key: value for key, value in row._mapping.items()})
                    except Exception:
                        # Fallback to a safer approach if _mapping doesn't work
                        try:
                            sample_data.append(dict(row))
                        except Exception:
                            pass  # Skip this row if we can't convert it
        except Exception as e:
            logger.error(f"Error getting sample data for {table_name}: {e}")
        
        # Infer table purpose
        purpose = infer_table_purpose(table_name, columns, sample_data)
        
        # Build column descriptions with data type and constraints
        column_desc = []
        for col in columns:
            constraints = []
            if col['name'] in pks.get('constrained_columns', []):
                constraints.append('PRIMARY KEY')
            nullable_str = '' if col['nullable'] else 'NOT NULL'
            
            # Add sample values for better context
            sample_values = []
            if sample_data:
                for row in sample_data[:3]:  # Use up to 3 sample rows
                    if col['name'] in row and row[col['name']]:
                        sample_values.append(str(row[col['name']]))
            
            sample_str = ""
            if sample_values:
                sample_str = f" (examples: {', '.join(sample_values[:3])})"
            
            column_desc.append(
                f"- {col['name']} ({col['type']}) {nullable_str} {' '.join(constraints)}{sample_str}"
            )
            
        # Build foreign key descriptions
        fk_desc = []
        for fk in fks:
            fk_desc.append(
                f"- {', '.join(fk['constrained_columns'])} -> {fk['referred_table']}.{', '.join(fk['referred_columns'])}"
            )
            
        # Add table description
        table_info.append(
            f"Table: {table_name}\n"
            f"Purpose: {purpose}\n"
            f"Columns:\n{''.join(f'{col}\n' for col in column_desc)}"
            f"Foreign Keys:\n{''.join(f'{fk}\n' for fk in fk_desc) if fk_desc else '- None\n'}"
            f"\n"
        )
    
    return '\n'.join(table_info)

def get_column_mappings(db: SQLDatabase) -> Dict[str, List[str]]:
    """
    Create mappings of common question topics to relevant columns across tables.
    This helps with query generation by identifying which columns contain certain types of information.
    """
    engine = db._engine
    inspector = inspect(engine)
    mappings = {}
    
    # Define common information categories
    categories = {
        "menu_items": ["product", "item", "dish", "food", "menu"],
        "contact_info": ["phone", "email", "contact", "address"],
        "hours": ["hour", "time", "schedule", "open", "close"],
        "location": ["location", "address", "place", "where"],
        "pricing": ["price", "cost", "rate", "fee"],
        "dietary": ["vegetarian", "vegan", "allergy", "gluten", "dairy"]
    }
    
    # For each table, analyze columns
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        
        for col in columns:
            col_name = col["name"].lower()
            
            # Map column to appropriate categories
            for category, keywords in categories.items():
                if any(keyword in col_name for keyword in keywords):
                    if category not in mappings:
                        mappings[category] = []
                    mappings[category].append(f"{table_name}.{col['name']}")
    
    return mappings