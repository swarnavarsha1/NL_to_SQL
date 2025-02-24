from typing import Dict, List
from sqlalchemy import create_engine, inspect, text
import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase
import os
from dotenv import load_dotenv

load_dotenv()

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
            print(f"Error getting sample data for {table_name}: {e}")
        
        schema_info[table_name] = {
            "columns": [{"name": col["name"], "type": str(col["type"])} for col in columns],
            "sample_data": sample_data,
            "relationships": inspector.get_foreign_keys(table_name)
        }
    
    return schema_info

def get_table_info(db: SQLDatabase) -> str:
    """Get detailed information about all tables in the database."""
    engine = db._engine
    inspector = inspect(engine)
    table_info = []
    
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        fks = inspector.get_foreign_keys(table_name)
        pks = inspector.get_pk_constraint(table_name)
        
        # Build column descriptions
        column_desc = []
        for col in columns:
            constraints = []
            if col['name'] in pks.get('constrained_columns', []):
                constraints.append('PRIMARY KEY')
            nullable_str = '' if col['nullable'] else 'NOT NULL'
            column_desc.append(
                f"- {col['name']} ({col['type']}) {nullable_str} {' '.join(constraints)}"
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
            f"Purpose: Stores {table_name.replace('_', ' ')} information\n"
            f"Columns:\n{''.join(f'{col}\n' for col in column_desc)}"
            f"Foreign Keys:\n{''.join(f'{fk}\n' for fk in fk_desc) if fk_desc else '- None\n'}"
            f"\n"
        )
    
    return '\n'.join(table_info)