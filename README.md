# Natural Language to SQL Chatbot

This project is a Streamlit-based chatbot that converts natural language queries into SQL queries and retrieves relevant data from a database. It utilizes LangChain, OpenAI GPT models, and SQLAlchemy to generate and execute SQL queries dynamically.

## Features
- Converts natural language questions into SQL queries.
- Retrieves relevant database tables based on user queries.
- Executes generated SQL queries and formats the output into human-readable responses.
- Displays available database tables in the sidebar.
- Uses OpenAI GPT models for query generation and response formulation.

## Project Structure

```sh
|-- main.py               # Streamlit app entry point
|-- db_utils.py           # Database connection and schema retrieval
|-- langchain_utils.py    # LangChain utilities for query generation and execution
|-- table_selection.py    # Table selection using LLM-based extraction
|-- prompts.py            # Templates for SQL query and response generation
|-- .env                  # Environment variables
```

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/swarnavarsha1/NL_to_SQL.git
   cd nl_to_sql

2. Create a virtual environment and activate it
    ```sh 
    python -m venv venv
    venv\Scripts\activate 

3. Install dependencies:
    ```sh
    pip install -r requirements.txt

4. Set up environment variables:<br>
    ```sh
    Create a .env file with the following
    db_user="user_name"
    db_password="password"
    db_host="localhost"
    db_name="database_name"
    OPENAI_API_KEY="api_key"

## Usage

Run the Streamlit application:
    ```sh
    streamlit run main.py


## Configuration
### Database Connection

The application connects to a MySQL database using SQLAlchemy.
The database credentials should be provided in the .env file.

### OpenAI API Key

The chatbot uses OpenAI's GPT model for SQL generation.
Provide your OpenAI API key in the .env file.

### How It Works

1. Table Selection (table_selection.py): Identifies relevant database tables based on the user query.
2. SQL Query Generation (langchain_utils.py): Uses GPT-4 to generate SQL queries based on the database schema.
3. Query Execution (db_utils.py): Runs the generated SQL query against the database and retrieves results.
4. Response Generation (langchain_utils.py): Formats the query results into a natural, conversational response.