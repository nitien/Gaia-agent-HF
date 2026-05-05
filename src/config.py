"""
Configuration and constants for the GAIA agent.
Centralized configuration for easy management and customization.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==================== API KEYS ====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY","")
# ==================== LLM CONFIGURATION ====================
#LLM_MODEL = "inclusionai/Ling-2.6-1T:free"
LLM_MODEL = "stepfun-ai/step-3.5-flash"
NVIDIA = 1
NVIDIA_MODEL="qwen/qwen3-coder-480b-a35b-instruct" 
LLM_TEMPERATURE = 0
LLM_MAX_ITERATIONS = 5

# ==================== TOOL CONFIGURATION ====================
WIKIPEDIA_MAX_PAGES = 2
WIKIPEDIA_CHAR_LIMIT = 8_000

YOUTUBE_CHAR_LIMIT = 10_000

WEB_SEARCH_RESULTS_LIMIT = 3

EXCEL_PREVIEW_ROWS = 50

# ==================== OUTPUT CONFIGURATION ====================
OUTPUT_FILE = "/home/nitin/AI/hfagent/results.jsonl"
FINAL_ANSWER_MAX_LENGTH = 100
REASONING_TRACE_MAX_LENGTH = 200

# ==================== TOOL NAMES ====================
TOOL_NAMES = {
    "WEB_SEARCH": "web_search",
    "WIKI_SEARCH": "wikisearch",
    "YOUTUBE_TRANSCRIPT": "youtube_transcript",
    "EXCEL_ANALYSIS": "load_and_analyze_excel_file",
    "IMAGE_TEXT": "extract_text_from_image",
    "AUDIO_TRANSCRIBE": "transcribe_audio",
    "ADD": "addition_tool",
    "SUBTRACT": "subtraction_tool",
    "MULTIPLY": "multiplication_tool",
    "NONE": "none",
}

# ==================== VALIDATION ====================
VALID_EXCEL_EXTENSIONS = (".xlsx", ".xls", ".csv")
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
VALID_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
