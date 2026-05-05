"""
Prompts for the GAIA agent workflow.
All prompts are centralized here for easy management and version control.
"""

PLANNER_PROMPT_TEMPLATE = """
You are a planning engine.follow the instructions strictly to Break the question into minimal executable steps.
and tools to perform each step.
Question: {question}

Return ONLY JSON that matches the required schema. No markdown, tags, or extra text.
Rules:
1) Use the fewest steps needed.
2) Use tool "none" only when no external tool is required.
3) If tool is 'load_and_analyze_excel_file', tool_input MUST be "query|/absolute/path/to/file.csv".
4) For math tools addition_tool/subtraction_tool/multiplication_tool, use "a,b".
5) Keep descriptions short and action-oriented.

"""

FINALIZER_PROMPT_TEMPLATE = """
Produce the final answer from the question and intermediate results.
Return ONLY the final answer text (no explanation).

Formatting rules:
- Number: no commas, no units unless explicitly requested.
- String: minimal words, no article unless required.
- Comma-separated list: sort alphabetically when the question expects a list.
- Respect required decimal precision.

Question: {question}
Intermediate results:
{intermediate_results}
"""

EXCEL_ANALYSIS_PROMPT_TEMPLATE = """
You are a data analyst. Analyze the following data and answer the question.
Data Summary:
{data_summary}
User Question: {query}
Provide a clear, concise answer based on the data provided. If you need to perform calculations, do them based on the data shown.
"""

WEB_SEARCH_EXTRACTION_PROMPT_TEMPLATE = """
From these search results: {search_results}
Extract the most relevant information that answers the query: '{query}'
Return a concise, factual answer.
"""

