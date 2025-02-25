from langchain_core.prompts import PromptTemplate

# SQL generation prompt with improved guidance and examples
sql_prompt = PromptTemplate.from_template(
    """You are an expert in converting natural language questions to SQL queries.
    
Database Schema:
{table_info}

Guidelines for query generation:
1. Analyze the question carefully to determine which table contains the relevant information
2. Use LIKE with wildcards (%) for text searches to be flexible with variations and partial matches
   - Example: WHERE column_name LIKE '%search_term%' instead of exact matches
3. Consider multiple potential columns that might contain relevant information
4. For text searches, consider both exact and partial matches
5. If the question refers to categories, times, or availability, check all relevant columns
6. When dealing with hyphenated terms or variations, use wildcards to match both formats
   - Example: LIKE '%north%indian%' will match both "north indian" and "north-indian"

Example Query Patterns:
1. For questions about "when" or "hours" → Look for time/hour related columns or questions
2. For questions about items or products → Search in name, description, and category columns
3. For questions about location or "where" → Look for address or location related columns
4. For questions about recommendations or "best" → Look for popular or recommended items
5. For specific attributes (sweet, spicy, vegetarian) → Search in relevant attribute columns

User Question: {question}

First, analyze which table(s) are needed, then identify appropriate columns and filtering conditions.
Generate a SQL query with flexible text matching to maximize the chance of finding relevant results.

SQL Query (return only the SQL, no other text): """)

# Answer prompt with better fallback handling and response structure
answer_prompt = PromptTemplate.from_template(
    """Given the following information, provide a natural, contextual response:
    
Original Question: {question}
SQL Query Used: {query}
Raw Query Result: {result}

Guidelines for response:
1. Provide a friendly, conversational response that directly answers the question
2. Format your response using these principles:
   - For lists of items (dishes, beverages, options), use bullet points (•) for clarity
   - For single item descriptions, use concise paragraphs
   - For information about times, address, or policies, use short, clear sentences
   - Use 1-2 sentences of introduction before lists
   - Keep descriptions brief but informative (2-3 sentences per item maximum)

3. For different types of content:
   - MENU ITEMS: Present as a bulleted list with item name in bold, followed by a brief description
   - HOURS/TIMINGS: Present clearly at the beginning of your response
   - LOCATIONS/CONTACT: Structure information in order of importance
   - SERVICES: Present as a bulleted list if multiple options

4. If the result is empty, suggest alternative ways to ask the question:
   - For product/item questions, suggest checking broader categories or similar terms
   - For time/availability questions, suggest asking in different ways
   - For location/service questions, suggest asking more specific aspects

5. Example format for a food item list:
   "Here are some [category] options you might enjoy:
   
   • **[Item Name]**: [Brief 1-2 sentence description]
   • **[Item Name]**: [Brief 1-2 sentence description]"

Response: """)

# Table selection prompt - generalized for any domain
table_selection_prompt = PromptTemplate.from_template(
    """Given a user's question, determine which database tables are most relevant.

Available Tables and Their Purposes:
{table_info}

User Question: {question}

Think about:
1. What type of information is the user looking for?
2. Which tables contain attributes, descriptions, or data that would answer this question?
3. Consider both direct and indirect matches - the relevant tables may not contain exact keywords

Relevant Tables: """)