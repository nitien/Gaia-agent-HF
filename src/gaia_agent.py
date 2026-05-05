"""
GAIA Agent - Multi-step reasoning agent for complex tasks.
Uses LanggraphStateGraph for workflow orchestration and multiple specialized tools.
"""

import os
import json
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.cache import SQLiteCache
from typing import TypedDict
import langchain_core
#from langchain_ollama import ChatOllama

CACHE_DB_PATH = ".langchain_llm_cache.db"
#langchain_core.globals.set_debug(True)
langchain_core.globals.set_llm_cache(SQLiteCache(database_path=CACHE_DB_PATH))
from customtools import (
    load_and_analyze_excel_file,
    extract_text_from_image,
    web_search,
    wikisearch,
    youtube_transcript,
    addition_tool,
    subtraction_tool,
    multiplication_tool,
    #transcribe_audio,
    modulus_tool,
    power_tool,
)
from config import (
    OPENROUTER_API_KEY,
    LLM_MODEL,
    NVIDIA,
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
    LLM_TEMPERATURE,
    OUTPUT_FILE,
    FINAL_ANSWER_MAX_LENGTH,
    REASONING_TRACE_MAX_LENGTH,
)
from prompts import (
    PLANNER_PROMPT_TEMPLATE,
    FINALIZER_PROMPT_TEMPLATE,
)

load_dotenv()
print(f"LangChain LLM cache enabled: {CACHE_DB_PATH}")

memory = MemorySaver()
llm = None
planner_llm = None


def connect_models():
    """Initialize and return the LLM instance."""
    try:
        global llm, planner_llm
        # llm = ChatOllama(
        #   model="gemma4:e2b",
        #   base_url="http://localhost:11434/",
        #   temperature=0,
        # )
        # Reset derived clients whenever base model is reconnected.
        planner_llm = None
        if NVIDIA:
             llm = ChatNVIDIA(
                             model=NVIDIA_MODEL,
                             api_key= NVIDIA_API_KEY, 
                             temperature=0.1,
                             top_p=1,
                            
             )
        else:
             print(f"Connecting to LLM: {LLM_MODEL}")
             llm = ChatOpenRouter(
                 model=LLM_MODEL,
                 temperature=LLM_TEMPERATURE,
                 api_key=OPENROUTER_API_KEY,
             )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise


# Tool registry
TOOLS = {
    "web_search": web_search,
    "addition_tool": addition_tool,
    "subtraction_tool": subtraction_tool,
    "multiplication_tool": multiplication_tool,
    "youtube_transcript": youtube_transcript,
    "load_and_analyze_excel_file": load_and_analyze_excel_file,
    "extract_text_from_image": extract_text_from_image,
    "wikisearch": wikisearch,
    #"transcribe_audio": transcribe_audio,
    "modulus_tool": modulus_tool,
    "power_tool":power_tool,
}


class AgentState(TypedDict):
    """State structure for the agent workflow."""
    question: str
    plan: List[Dict[str, Any]]
    current_step: int
    selected_tool: Optional[str]
    tool_input: Optional[str]
    tool_output: Optional[str]
    intermediate_results: List[Dict[str, Any]]
    final_answer: Optional[str]
    done: bool


class Step(BaseModel):
    """Represents a single step in the plan."""
    step_number: int
    description: str
    tool: Literal[
        "web_search",
        "wikisearch",
        "youtube_transcript",
        "load_and_analyze_excel_file",
        "extract_text_from_image",
         #"transcribe_audio",
        "addition_tool",
        "subtraction_tool",
        "multiplication_tool",
        "modulus_tool",
        "power_tool",
        "none",
    ]
    tool_input: str


class Plan(BaseModel):
    """Structured plan with multiple steps."""
    steps: List[Step]


def get_planner_llm():
    """Create structured planner client once and reuse it across questions."""
    global planner_llm, llm
    if llm is None:
        llm = connect_models()
    if planner_llm is None:
        planner_llm = llm.with_structured_output(Plan, method="json_schema")
    return planner_llm


def planner_node(state: AgentState):
    """Planner node: breaks down question into steps."""
    prompt = PLANNER_PROMPT_TEMPLATE.format(question=state['question'])

    response = get_planner_llm().invoke(prompt)
    
    print(f"Planner generated {len(response.steps)} steps")
    
    return {
        **state,
        "plan": [step.model_dump() for step in response.steps],
        "current_step": 0,
        "intermediate_results": [],
        "done": False,
    }



def execute_step_node(state: AgentState):
    """Execute step node: prepares tool invocation."""
    step = state["plan"][state["current_step"]]
    print(f"Current Step:{step}")
    tool_name = step.get("tool", "none")
    
    print(f"Executing step {state['current_step'] + 1}/{len(state['plan'])}: {tool_name}")
    
    return {
        **state,
        "tool_input": step.get("tool_input"),
        "selected_tool": tool_name,
    }


def tool_node(state: AgentState):
    """Tool execution node: invokes the selected tool."""
    tool_name = state.get("selected_tool")
    tool_input = state.get("tool_input")

    if tool_name == "none":
        return {**state, "tool_output": tool_input}
    
    print(f"Invoking tool: {tool_name}")
    tool = TOOLS.get(tool_name)
    
    # Special handling for load_and_analyze_excel_file: parse query|file_path format
    if tool_name == "load_and_analyze_excel_file" and isinstance(tool_input, str) and "|" in tool_input:
        parts = tool_input.split("|", 1)
        query = parts[0].strip()
        file_path = parts[1].strip()
        tool_input = {"query": query, "file_path": file_path}
        print(f"Parsed Excel input - Query: '{query[:50]}...', File: '{file_path}'")
    
    # Special handling for math tools: parse "a,b" format
    if tool in (addition_tool, subtraction_tool, multiplication_tool):
        try:
            a, b = tool_input.split(",")
            tool_input = {"a": a.strip(), "b": b.strip()}
        except Exception as e:
            print(f"Error parsing math tool input: {e}")
            return {**state, "tool_output": f"Error parsing input: {e}"}
    
    if not tool:
        return {**state, "tool_output": f"Unknown tool: {tool_name}"}

    try:
        result = tool.invoke(tool_input)
    except Exception as e:
        print(f"Error invoking tool {tool_name}: {e}")
        result = f"Tool error: {str(e)}"

    return {**state, "tool_output": result}


def update_state_node(state: AgentState):
    """Update state node: records tool output and progresses to next step."""
    step = state["plan"][state["current_step"]]

    state["intermediate_results"].append({
        "step": step,
        "output": state["tool_output"]
    })

    next_step = state["current_step"] + 1
    done = next_step >= len(state["plan"])

    return {
        **state,
        "current_step": next_step,
        "done": done,
    }




def should_continue(state: AgentState):
    """Conditional edge: determines if workflow should continue or finalize."""
    return "finalize" if state["done"] else "continue"


def finalizer_node(state: AgentState):
    """Finalizer node: summarizes results and generates final answer."""
    # Compact context to reduce token usage sent to finalizer.
    results_text = "\n".join([
        f"S{i+1}: {r['step'].get('description', '')} | O: {str(r['output'])[:80]}"
        for i, r in enumerate(state["intermediate_results"])
    ])
    
    prompt = FINALIZER_PROMPT_TEMPLATE.format(
        question=state['question'],
        intermediate_results=results_text
    )
    
    response = llm.invoke(prompt)

    return {
        **state,
        "final_answer": response.content,
    }



def create_agent_workflow():

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", execute_step_node)
    graph.add_node("tool", tool_node)
    graph.add_node("updater", update_state_node)
    graph.add_node("finalizer", finalizer_node) 
    # Entry
    graph.set_entry_point("planner")

    # Flow
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "tool")
    graph.add_edge("tool", "updater")

    # Loop
    graph.add_conditional_edges(
        "updater",
        should_continue,
        {
            "continue": "executor",
            "finalize": "finalizer"
        }
    )

    # End
    graph.add_edge("finalizer", END)

    return graph.compile()


def format_reasoning_trace(intermediate_results: List[Dict[str, Any]]) -> str:
    """Format intermediate results into a readable reasoning trace"""
    trace_lines = []
    for result in intermediate_results:
        step = result.get("step", {})
        output = result.get("output", "")
        description = step.get("description", "Unknown step")
        tool = step.get("tool", "none")
        
        trace_lines.append(f"Step: {description}")
        trace_lines.append(f" Tool: {tool}")
        trace_lines.append(f" Output: {output[:200]}{'...' if len(str(output)) > 200 else ''}")
    
    return "\n".join(trace_lines)


def process_questions(questions_file: str = None, questions_list: List[str] = None) -> str:
    """
    Process multiple questions and save results to a file
    
    Args:
        questions_file: Path to a file containing questions (one per line)
        questions_list: List of questions to process
    
    Returns:
        Path to the output file with results
    """
    global llm
    llm = connect_models()
    print(f"LLM available: {llm}")
    agent = create_agent_workflow()
    
    # Get questions from either file or list
    if questions_file:
        with open(questions_file, 'r') as f:
            questions = [q.strip() for q in f.readlines() if q.strip()]
    elif questions_list:
        questions = questions_list
    else:
        raise ValueError("Either questions_file or questions_list must be provided")
    
    results = []
    
    for idx, question in enumerate(questions, 1):
        task_id = f"task_id_{idx}"
        print(f"\n{'='*80}")
        print(f"Processing {task_id}: {question[:80]}...")
        print(f"{'='*80}")
        
        try:
            # Run the agent
            result = agent.invoke({
                "question": question
            })
            
            # Extract the final answer and reasoning trace
            final_answer = result.get("final_answer", "No answer generated")
            intermediate_results = result.get("intermediate_results", [])
            
            # Format the reasoning trace
            reasoning_trace = format_reasoning_trace(intermediate_results)
            
            # Create the result object
            task_result = {
                "task_id": task_id,
                "model_answer": final_answer,
                "reasoning_trace": reasoning_trace
            }
            
            results.append(task_result)
            
            print(f"Completed {task_id}")
            print(f"Answer: {final_answer[:100]}...")
            
        except Exception as e:
            print(f"✗ Error processing {task_id}: {str(e)}")
            task_result = {
                "task_id": task_id,
                "model_answer": f"Error: {str(e)}",
                "reasoning_trace": "Failed to execute agent"
            }
            results.append(task_result)
    
    # Save results to file
    output_file = "/home/nitin/AI/hfagent/results.jsonl"
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n{'='*80}")
    print(f"All tasks completed. Results saved to: {output_file}")
    print(f"{'='*80}")
    
    return output_file





if __name__ == "__main__":
    # Example questions to process
    questions = [
        """
          Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations.
        """  
        #Task ID: 52e8ce1c-09bd-4537-8e2d-67d1648779b9 ; Question: The attached .csv file shows precipitation amounts, in inches, for the five boroughs of New York City in a certain year. How many inches of precipitation did the city receive in total for that year? Don’t use commas if the number has four or more digits. ; file_name: /home/nitin/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA/snapshots/682dd723ee1e1697e00360edccf2366dc8418dd9/2023/test/52e8ce1c-09bd-4537-8e2d-67d1648779b9.csv

        #"What is the square of the population of France in millions?",
        #"What is 50 plus 75?"
    ]
    
    # Process all questions
    output_file = process_questions(questions_list=questions)
    
    # Print the results
    print("\nResults from file:")
    with open(output_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            print(f"\nTask ID: {result['task_id']}")
            print(f"Answer: {result['model_answer']}")
            print(f"Reasoning:\n{result['reasoning_trace']}") 
