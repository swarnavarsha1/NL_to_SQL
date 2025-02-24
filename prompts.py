from langchain_core.prompts import PromptTemplate

# SQL generation prompt
sql_prompt = PromptTemplate.from_template(
    """You are an expert in converting natural language questions to SQL queries. 
    
Database Schema:
{table_info}

Guidelines for query generation:
1. Understand the intent behind the question, not just keyword matching
2. Use appropriate wildcards (%) for flexible text matching
3. Consider synonyms and related terms
4. Join relevant tables when needed
5. For text searches, consider partial matches and variations

For example:
- "where is it?" -> location/address related queries
- "what time" -> hours/timing related queries
- "how much" -> price/cost related queries
- "can I" -> availability/service related queries

User Question: {question}
First, identify the intent: Think about what information the user is really looking for.
Then, generate a SQL query that will best answer this intent.

SQL Query (return only the SQL, no other text): """)

# Answer prompt
answer_prompt = PromptTemplate.from_template(
    """Given the following information, provide a natural, contextual response:
    
Original Question: {question}
SQL Query Used: {query}
Raw Query Result: {result}

Guidelines for response:
1. Provide a natural, conversational response
2. Include context and relevant details
3. If multiple related pieces of information are available, combine them logically
4. If the result is empty, suggest alternative phrasings or related information
5. Make the response sound natural, not like you're reading from a database

Response: """)

# Table selection prompt for future use with multiple tables
table_selection_prompt = PromptTemplate.from_template(
    """Given a user's question, determine which database tables are most relevant.

Available Tables and Their Purposes:
{table_info}

User Question: {question}

Think about:
1. What information is the user really looking for?
2. Which tables might contain this information?
3. Are there related tables that might have relevant context?

Relevant Tables: """)