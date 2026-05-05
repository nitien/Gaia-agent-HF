"""Sequential Task Executor agent scaffold.

This module provides a lightweight agent structure that:
- accepts a user prompt,
- analyzes it into ordered tasks,
- executes tasks sequentially (tools can be registered),
- passes structured output from one task to the next when needed,
logs actions and errors for easy tracing.

It is intentionally generic: tools are simple callables that receive a
params dict and return a dict. The planner uses an LLM if available to
produce a JSON plan; otherwise a deterministic fallback is used.
"""
import langchain
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import logging
import traceback
import json
import os

# Load .env early so connect_llm can pick up tokens
try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	# dotenv is optional; environment variables may already be set
	pass


# Module logger
logger = logging.getLogger("hfagent.agent")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
ch.setFormatter(formatter)
if not logger.handlers:
	logger.addHandler(ch)


# Optional integrations
has_wikipedia = False
has_ddg = False
has_langchain = False

try:
	import wikipedia
	has_wikipedia = True
except Exception:
	has_wikipedia = False

try:
	# duckduckgo_search package exports vary by version; try common names
	try:
		from duckduckgo_search import ddg  # type: ignore
		has_ddg = True
	except Exception:
		from duckduckgo_search import DDGS  # type: ignore
		has_ddg = True
except Exception:
	has_ddg = False

try:
	import langchain  # type: ignore
	has_langchain = True
except Exception:
	has_langchain = False

# Global LLM instance (set via connect_llm)
_LLM: Optional[Any] = None


def connect_llm(backend: str = "gemini", **cfg) -> Any:
	"""Create and return a LangChain LLM/chat model instance.

	Example usage:
	  connect_llm("hf", repo_id="google/flan-t5-xl", huggingfacehub_api_token=...)
	  connect_llm("gemini", model_name="gemini-proto")

	This function expects LangChain to be installed and configured.
	It stores the instance in a module-global variable for llm_call to use.
	"""
	global _LLM
	# Fill cfg from environment if not explicitly provided
	if backend == "gemini":
		# Gemini/Google chat model via LangChain
		api_key = cfg.get("api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
		model_name = cfg.get("model_name") or os.getenv("GEMINI_MODEL")
		from langchain.chat_models import ChatGoogleGemini  # type: ignore
		if api_key:
			os.environ.setdefault("GOOGLE_API_KEY", api_key)
		_LLM = ChatGoogleGemini(model_name=model_name) if model_name else ChatGoogleGemini()
	elif backend == "hf":
		# HuggingFace Hub wrapper
		repo_id = cfg.get("repo_id") or os.getenv("HUGGINGFACEHUB_REPO") or os.getenv("HUGGINGFACEHUB_MODEL")
		token = cfg.get("huggingfacehub_api_token") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_KEY")
		from langchain import HuggingFaceHub  # type: ignore
		if token:
			os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
			_LLM = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=token)
		else:
			_LLM = HuggingFaceHub(repo_id=repo_id)
	else:
		# Default: ChatOpenAI-like wrapper; prefer OPENAI_API_KEY from env
		model_name = cfg.get("model_name") or os.getenv("OPENAI_MODEL")
		api_key = cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
		from langchain.chat_models import ChatOpenAI  # type: ignore
		if api_key:
			os.environ.setdefault("OPENAI_API_KEY", api_key)
		if model_name:
			_LLM = ChatOpenAI(temperature=cfg.get("temperature", 0), model_name=model_name)
		else:
			_LLM = ChatOpenAI(temperature=cfg.get("temperature", 0))

	return _LLM

# Try to import LangChain tool decorator (optional)
has_lc_tool = False
try:
	from langchain.tools import tool as lc_tool  # type: ignore
	has_lc_tool = True
except Exception:
	has_lc_tool = False


def llm_call(prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
	"""Lightweight LLM abstraction.

	Tries a few common libraries. Returns {"text": str} on success or
	{"error": ...} on failure. Replace with your project's LLM wrapper
	for production use.
	"""
	# Use a globally-connected LangChain LLM if available
	global _LLM
	if _LLM is None:
		_LLM = connect_llm("gemini", model_name="gemini-2.5-pro")
		# Prefer an explicit call to connect_llm(). If the user did not
		# call it, avoid importing LiC-dependent defaults which can vary
		# across LangChain versions. Return a deterministic fallback.
		#logger.info("No LLM connected via connect_llm(); returning prompt fallback")
		#return {"text": prompt[:200]}

	# Call the LangChain LLM/chat model
	out = _LLM(prompt)
	# LangChain chat models may return a string or an object; normalize to text
	try:
		return {"text": str(out)}
	except Exception:
		return {"text": repr(out)}


@dataclass
class Task:
	name: str
	params: Dict[str, Any] = field(default_factory=dict)
	result: Optional[Any] = None


class TaskExecutor:
	"""Executes a list of Tasks sequentially, passing outputs forward.

	Task.params can include:
	- tool: name of registered tool to call
	- func: a local callable to run
	- input: explicit input dict for the tool
	- input_from_previous: key name in previous result to pass as 'input'
	"""

	def __init__(self):
		self.tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

	def register_tool(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]):
		logger.debug("Registering tool: %s", name)
		self.tools[name] = func

	def call_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
		if name not in self.tools:
			err = {"error": f"Tool not registered: {name}"}
			logger.error(err["error"])
			return err
		try:
			logger.info("Calling tool %s", name)
			out = self.tools[name](params)
			# If a LangChain-decorated tool returns a JSON string, try to decode it
			if isinstance(out, str):
				try:
					parsed = json.loads(out)
					logger.debug("Tool %s returned JSON-string, parsed keys: %s", name, list(parsed.keys()) if isinstance(parsed, dict) else type(parsed))
					return parsed
				except Exception:
					# not JSON, return as text under 'text' key
					return {"text": out}
			logger.debug("Tool %s output keys: %s", name, list(out.keys()) if isinstance(out, dict) else type(out))
			return out
		except Exception as e:
			logger.error("Tool %s failed: %s", name, e)
			logger.debug(traceback.format_exc())
			return {"error": str(e)}

	def execute_tasks(self, tasks: List[Task]) -> List[Task]:
		logger.info("Executing %d tasks", len(tasks))
		prev_result: Optional[Dict[str, Any]] = None
		for i, task in enumerate(tasks):
			logger.info("Task %d/%d: %s", i + 1, len(tasks), task.name)
			try:
				# Build input for this task
				input_data = task.params.get("input")
				if input_data is None and task.params.get("input_from_previous") and prev_result is not None:
					key = task.params.get("input_from_previous")
					input_data = prev_result.get(key, prev_result)
					# If previous result returned a raw string (e.g., page text),
					# wrap it into a dict expected by tools like llm_extract.
					if not isinstance(input_data, dict):
						input_data = {"text": input_data}

				tool_name = task.params.get("tool")
				if tool_name:
					res = self.call_tool(tool_name, input_data or {})
				else:
					func = task.params.get("func")
					if callable(func):
						res = func(input_data or {})
					else:
						res = {"result": None}

				task.result = res
				# set prev_result for next tasks; prefer dicts
				if isinstance(res, dict):
					prev_result = res
				else:
					prev_result = {"result": res}

			except Exception as e:
				logger.error("Task %s exception: %s", task.name, e)
				logger.debug(traceback.format_exc())
				task.result = {"error": str(e)}
				prev_result = task.result

		logger.info("Task execution finished")
		return tasks


# --- Generic tools -------------------------------------------------

def tool_wikipedia_page(params: Dict[str, Any]) -> Dict[str, Any]:
	"""Fetches the best Wikipedia page for a query and returns plain text.

	Returns {"text": ..., "title": ..., "url": ...} or {"error": ...}
	"""
	query = params.get("query")
	if not query:
		return {"error": "missing query"}

	if not has_wikipedia:
		return {"error": "wikipedia package not installed"}

	try:
		results = wikipedia.search(query)
		if not results:
			return {"error": "no wikipedia results"}
		title = results[0]
		page = wikipedia.page(title)
		return {"title": page.title, "url": page.url, "text": page.content}
	except Exception as e:
		logger.error("wikipedia fetch failed: %s", e)
		return {"error": str(e)}


def tool_web_search(params: Dict[str, Any]) -> Dict[str, Any]:
	"""Perform a web search (DuckDuckGo) and return a list of results.

	Returns {"results": [{"title":..., "href":..., "body":...}, ...]}
	"""
	query = params.get("query")
	if not query:
		return {"error": "missing query"}

	if not has_ddg:
		return {"error": "duckduckgo_search not available"}

	try:
		# ddg may be a function or require DDGS context depending on version
		try:
			from duckduckgo_search import ddg  # type: ignore
			raw = ddg(query, max_results=5)
			# ddg returns list of dicts with title, href, body
			return {"results": raw}
		except Exception:
			from duckduckgo_search import DDGS  # type: ignore
			results = []
			with DDGS() as ddgs:
				for r in ddgs.text(query, max_results=5):
					results.append(dict(r))
			return {"results": results}
	except Exception as e:
		logger.error("web search failed: %s", e)
		return {"error": str(e)}


def tool_llm_extract(params: Dict[str, Any]) -> Dict[str, Any]:
	"""Ask the LLM to extract structured information from text.

	Expected params: {"text": ..., "instruction": ...}
	Returns {"extracted": ...}
	"""
	text = params.get("text")
	instruction = params.get("instruction") or "Extract the requested information from the input text."
	if not text:
		return {"error": "missing text to extract from"}

	prompt = f"Instruction:\n{instruction}\n\nInput:\n{text}\n\nRespond with the extracted information in JSON." 
	out = llm_call(prompt)
	return {"extracted": out.get("text")}


def tool_select_best(params: Dict[str, Any]) -> Dict[str, Any]:
	"""Use LLM to pick the best candidate result from a list.

	Expected params: {"candidates": [...], "instruction": str}
	Returns {"best": ...}
	"""
	candidates = params.get("candidates")
	instruction = params.get("instruction", "Select the best candidate that answers the user's question and return it verbatim.")
	if not candidates:
		return {"error": "no candidates"}

	# Build a short prompt
	prompt = f"Instruction:\n{instruction}\n\nCandidates:\n{json.dumps(candidates) }\n\nReturn the single best candidate." 
	out = llm_call(prompt)
	return {"best": out.get("text")}


# --- Planner / Analyzer --------------------------------------------

def analyze_input_to_tasks(user_input: str) -> List[Task]:
	"""Produce a list of tasks for the executor.

	Strategy:
	1. Try to ask an LLM (via llm_call) to emit a JSON list of tasks.
	2. If that fails or LLM not available, fall back to a simple heuristic:
	   - Prefer wikipedia_page -> llm_extract
	   - Else web_search -> select_best -> llm_extract

	The JSON plan expected from LLM is a list of objects: {"name":..., "tool":..., "params": {...}}
	"""
	# Try LLM planner
	planner_prompt = (
		"You are a planner that turns a user question into a JSON array of tasks.\n"
		"Each task must be an object with: name (string), tool (string), params (object).\n"
		f"User question: {user_input}\n\n"
		"Return only valid JSON. Keep tasks short and focused."
	)

	plan_out = llm_call(planner_prompt)
	plan_text = plan_out.get("text", "")
	# Try to parse JSON
	try:
		plan = json.loads(plan_text)
		if isinstance(plan, list):
			tasks = [Task(name=p.get("name", f"task_{i}"), params=p.get("params", {})) for i, p in enumerate(plan)]
			logger.info("Planner produced %d tasks", len(tasks))
			return tasks
	except Exception:
		logger.debug("Planner did not return JSON; falling back (%s...)", plan_text[:120])

	# Fallback heuristic
	tasks: List[Task] = []
	if has_wikipedia:
		tasks.append(Task(name="fetch_wikipedia", params={"tool": "wikipedia_page", "input": {"query": user_input}}))
		tasks.append(Task(name="extract_info", params={"tool": "llm_extract", "input_from_previous": "text", "instruction": "Extract the core answer to the user's question from the page content."}))
	else:
		tasks.append(Task(name="web_search", params={"tool": "web_search", "input": {"query": user_input}}))
		tasks.append(Task(name="select_best", params={"tool": "select_best", "input_from_previous": "results", "instruction": "Select the best search result that answers the user's question."}))
		tasks.append(Task(name="extract_info", params={"tool": "llm_extract", "input_from_previous": "best", "instruction": "Extract the core answer from the selected page or snippet."}))

	return tasks


def main_agent(user_input: str) -> Dict[str, Any]:
	executor = TaskExecutor()

	# Register generic tools
	executor.register_tool("wikipedia_page", tool_wikipedia_page)
	executor.register_tool("web_search", tool_web_search)
	executor.register_tool("llm_extract", tool_llm_extract)
	executor.register_tool("select_best", tool_select_best)

	# If LangChain tool decorator is available, expose decorated wrappers
	# and prefer the decorated adapters (they call the same underlying impl).
	if has_lc_tool:
		try:
			@lc_tool
			def wikipedia_page_tool(query: str) -> str:
				# LangChain tools typically accept simple args; return JSON string
				return json.dumps(tool_wikipedia_page({"query": query}))

			@lc_tool
			def web_search_tool(query: str) -> str:
				return json.dumps(tool_web_search({"query": query}))

			@lc_tool
			def llm_extract_tool(payload: str) -> str:
				# payload is JSON string or plain text
				try:
					params = json.loads(payload)
				except Exception:
					params = {"text": payload}
				return json.dumps(tool_llm_extract(params))

			@lc_tool
			def select_best_tool(payload: str) -> str:
				try:
					params = json.loads(payload)
				except Exception:
					return json.dumps({"error": "invalid payload"})
				return json.dumps(tool_select_best(params))

			# Adapter functions: accept dict params, call decorated tool and parse result
			def _wikipedia_via_lc(p: Dict[str, Any]) -> Dict[str, Any]:
				q = p.get("query") if isinstance(p, dict) else p
				s = wikipedia_page_tool(q)
				try:
					return json.loads(s)
				except Exception:
					return {"text": s}

			def _websearch_via_lc(p: Dict[str, Any]) -> Dict[str, Any]:
				q = p.get("query") if isinstance(p, dict) else p
				s = web_search_tool(q)
				try:
					return json.loads(s)
				except Exception:
					return {"text": s}

			def _llm_extract_via_lc(p: Dict[str, Any]) -> Dict[str, Any]:
				# pass JSON string payload
				s = llm_extract_tool(json.dumps(p))
				try:
					return json.loads(s)
				except Exception:
					return {"text": s}

			def _select_best_via_lc(p: Dict[str, Any]) -> Dict[str, Any]:
				s = select_best_tool(json.dumps(p))
				try:
					return json.loads(s)
				except Exception:
					return {"text": s}

			# Override registration to prefer LangChain-backed implementations
			executor.register_tool("wikipedia_page", _wikipedia_via_lc)
			executor.register_tool("web_search", _websearch_via_lc)
			executor.register_tool("llm_extract", _llm_extract_via_lc)
			executor.register_tool("select_best", _select_best_via_lc)
			logger.info("Registered LangChain-decorated tool adapters")
		except Exception as e:
			logger.debug("Failed to register LangChain decorated tools: %s", e)

	tasks = analyze_input_to_tasks(user_input)
	executed = executor.execute_tasks(tasks)

	result = {t.name: t.result for t in executed}
	return result


if __name__ == "__main__":
	# Example: generic question — the agent will prefer Wikipedia if available
	sample_q = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?"
	logger.info("Running generic agent for: %s", sample_q)
	out = main_agent(sample_q)
	print(json.dumps(out, indent=2, ensure_ascii=False))
