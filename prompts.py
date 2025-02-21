from langchain_core.prompts import PromptTemplate

# SQL generation prompt
sql_prompt = PromptTemplate.from_template(
    """Given an input question, create a syntactically correct MySQL query.
    
    Use the following schema:
    {table_info}
    
    Question: {question}
    SQL: """
)

# Answer prompt
answer_prompt = PromptTemplate.from_template(
    """Based on the question and SQL query results, provide a natural language answer.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
)