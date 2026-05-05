"""
Custom tools for the GAIA agent.
Includes tools for web search, file analysis, text extraction, and more.
"""

import os
import re
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path

import cv2
import pandas as pds
import pytesseract
import whisper
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader
from langchain_openrouter import ChatOpenRouter
from tavily import TavilyClient
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_nvidia_ai_endpoints import ChatNVIDIA
global llm

from config import (
    OPENROUTER_API_KEY,
    NVIDIA_MODEL,
    NVIDIA,
    NVIDIA_API_KEY,
    TAVILY_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    WIKIPEDIA_MAX_PAGES,
    WIKIPEDIA_CHAR_LIMIT,
    YOUTUBE_CHAR_LIMIT,
    WEB_SEARCH_RESULTS_LIMIT,
    EXCEL_PREVIEW_ROWS,
)
from prompts import (
    EXCEL_ANALYSIS_PROMPT_TEMPLATE,
    WEB_SEARCH_EXTRACTION_PROMPT_TEMPLATE,
)

load_dotenv()
@tool
def wikisearch(query: str, max_pages: int = None) -> str:
    """Search Wikipedia pages and return concatenated page texts."""
    max_pages = max_pages or WIKIPEDIA_MAX_PAGES
    print(f"wikisearch called with query: {query}, max_pages: {max_pages}")
    
    try:
        docs = WikipediaLoader(query=query, load_max_docs=max_pages).load()
        joined = "\n\n---\n\n".join(d.page_content for d in docs)
        return joined[:WIKIPEDIA_CHAR_LIMIT]
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"


@tool
def youtube_transcript(url: str) -> str:
    """Fetch YouTube video transcript from given URL."""
    chars = YOUTUBE_CHAR_LIMIT
    video_id_match = re.search(r"[?&]v=([A-Za-z0-9_\-]{11})", url)
    #print(f"Video is {video_id_match.group(1)}")
    video_id = None
    if not video_id_match:
        return "Error: Could not extract video ID from URL"
    else:
        video_id=video_id_match.group(1)
    #print(YouTubeTranscriptApi.__getattribute__())
    #print(video_id)
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        text = "\n".join(piece.text for piece in transcript.snippets)
        return text[:chars]
    except Exception as exc:
        print(f"Error fetching YouTube transcript: {exc}")
        return f"Error fetching transcript: {str(exc)}"
    

@tool
def web_search(query: str) -> str:
    """Perform a web search and extract concise factual answers."""
    print(f"web_search called with query: {query}")
    
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY not set in environment"
    
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        search_results = tavily_client.search(query)
        print(f"Search results obtained")
        
        # Format results as a readable string
        if search_results and isinstance(search_results, dict) and "results" in search_results:
            formatted = "\n".join([
                f"- {r.get('title', '')}: {r.get('content', '')[:200]}"
                for r in search_results["results"][:WEB_SEARCH_RESULTS_LIMIT]
            ])
            return formatted if formatted else "No results found"
        
        return str(search_results)
    except Exception as e:
        print(f"Error during web search: {e}")
        return f"Error during web search: {str(e)}"

@tool
def addition_tool(a: str, b: str) -> str:
    """Add two numbers represented as strings."""
    try:
        num_a = float(a)
        num_b = float(b)
        result = num_a + num_b
        return str(result)
    except ValueError:
        return "Invalid input: both a and b must be numbers."
    except Exception as e:
        return f"Error during addition: {str(e)}"
    
@tool
def subtraction_tool(a: str, b: str) -> str:
    """Subtract two numbers represented as strings."""
    try:
        num_a = float(a)
        num_b = float(b)
        result = num_a - num_b
        return str(result)
    except ValueError:
        return "Invalid input: both a and b must be numbers."
    except Exception as e:
        return f"Error during subtraction: {str(e)}"
    

@tool
def multiplication_tool(a: str, b: str) -> str:
    """Multiply two numbers represented as strings."""
    try:
        num_a = float(a)
        num_b = float(b)
        result = num_a * num_b
        return str(result)
    except ValueError:
        return "Invalid input: both a and b must be numbers."
    except Exception as e:
        return f"Error during multiplication: {str(e)}"
    

@tool
def division_tool(a: str, b: str) -> str:
    """Divide two numbers represented as strings."""
    try:
        num_a = float(a)
        num_b = float(b)
        if num_b == 0:
            return "Error: Division by zero is not allowed."
        result = num_a / num_b
        return str(result)
    except ValueError:
        return "Invalid input: both a and b must be numbers."
    except Exception as e:
        return f"Error during division: {str(e)}"



@tool
def modulus_tool(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    result = a % b 
    return str(result)

@tool
def power_tool(a: float, b: float) -> float:
    """Get the power of two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    result = a**b
    return str(result)


@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from image files using OCR.
    Works with .jpg, .png, .bmp, .tiff formats only.
    
    Args:
        image_path: Full path to the image file
    """
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        thresh = cv2.bitwise_not(thresh)
        
        custom_config = r'--oem 3 --psm 6'
        full_text = pytesseract.image_to_string(thresh, config=custom_config)
        
        return f"Extracted text from image:\n\n{full_text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"



@tool
def run_python(code: str) -> str:
    """Execute Python code in a subprocess and return output."""
    try:
        with NamedTemporaryFile(delete=False, suffix=".py", mode="w") as f:
            f.write(code)
            path = f.name
        
        proc = subprocess.run(
            ["python", path], capture_output=True, text=True, timeout=45
        )
        
        out = proc.stdout.strip().splitlines()
        return out[-1] if out else ""
    except Exception as exc:
        print(f"Error executing Python code: {exc}")
        return f"py_error:{exc}"

@tool
def load_and_analyze_excel_file(query: str, file_path: str) -> str:
    """
    Load and analyze data from Excel/CSV files (.xlsx, .xls, .csv).
    
    Args:
        query: Data analysis question (e.g., "Count records where status=active")
        file_path: Full path to the Excel/CSV file
    """
    print(f"load_and_analyze_excel_file called - Query: {query}, File: {file_path}")
    
    try:
        # Read the file based on extension
        if file_path.lower().endswith(".csv"):
            df = pds.read_csv(file_path)
        else:
            df = pds.read_excel(file_path)
        
        # Create basic data summary
        result = f"File loaded successfully.\n"
        result += f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
        result += f"Column names: {', '.join(df.columns.tolist())}\n\n"
        
        # Prepare data context for LLM
        data_summary = f"DataFrame:\n{df.to_string(max_rows=EXCEL_PREVIEW_ROWS)}\n\nData Types:\n{df.dtypes.to_string()}"
        
        # Create analysis prompt
        analysis_prompt = EXCEL_ANALYSIS_PROMPT_TEMPLATE.format(
            data_summary=data_summary,
            query=query
        )
        tool_llm=None
        
        if NVIDIA:
            tool_llm = ChatNVIDIA(
                                model=NVIDIA_MODEL,
                                api_key= NVIDIA_API_KEY, 
                                temperature=LLM_TEMPERATURE,
                                top_p=1,
                                
            )
        else:        
            # Get LLM analysis
            tool_llm = ChatOpenRouter(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                api_key=OPENROUTER_API_KEY,
            )
        
        message = HumanMessage(content=analysis_prompt)
        llm_response = tool_llm.invoke([message])
        
        result += f"Analysis:\n{llm_response.content}"
        print(f"Excel analysis completed")
        return result

    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"



#@tool
# def transcribe_audio(audio_file: str) -> str:
#     """Transcribe audio files and return the transcript."""
#     try:
#         model = whisper.load_model("small")
#         output = model.transcribe(audio=str(Path(audio_file)), language='en')
#         print(f"Audio transcription completed")
#         return output['text']
#     except Exception as exc:
#         print(f"Error transcribing audio: {exc}")
#         return f"transcription_error:{exc}"

