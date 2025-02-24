import streamlit as st
from langchain_utils import invoke_chain
from dotenv import load_dotenv
import os
from db_utils import get_database, get_table_info
from sqlalchemy import inspect

load_dotenv()

def get_table_descriptions(db) -> dict:
    """Get descriptions of all tables in the database."""
    engine = db._engine
    inspector = inspect(engine)
    table_info = {}
    
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        column_info = [f"• {col['name']} ({col['type']})" for col in columns]
        table_info[table_name] = column_info
    
    return table_info

def display_sidebar_tables():
    """Display available tables in the sidebar."""
    st.sidebar.title("Database Schema")
    
    try:
        db = get_database()
        table_info = get_table_descriptions(db)
        
        st.sidebar.write("Available Tables:")
        
        # Create expandable sections for each table
        for table_name, columns in table_info.items():
            with st.sidebar.expander(f"📊 {table_name}"):
                st.write("Columns:")
                for column in columns:
                    st.write(column)
                    
    except Exception as e:
        st.sidebar.error("Unable to load database schema")

def main():
    st.title("Natural Language to SQL Chatbot")
    
    # Display tables in sidebar
    display_sidebar_tables()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            try:
                response = invoke_chain(prompt, st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = "I apologize, but I'm having trouble processing your request. Please try again."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()