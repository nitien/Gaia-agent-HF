# GAIA LangGraph Agent Project For HuggingFace Agent course Unit-4.

This project provides a basic structure for building agents using LangGraph.
It fetches 20 questions from GAIA-HF site , the agent processes the questions in steps , and provides the answer.


## Project Structure

```
hfagent/
├── src/                 # Source code
│   ├── __init__.py
│   └── agent.py         # Main agent implementation
├── tests/               # Test files
│   └── test_agent.py
├── docs/                # Documentation (empty for now)
├── venv/                # Virtual environment
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables
└── README.md            # This file
```

## Setup

1. Create virtual environment (already done):
   ```bash
   python3 -m venv venv
   ```

2. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   ```

## Usage

Run the agent:
```bash
source venv/bin/activate
python src/agent.py
```

Run tests:
```bash
source venv/bin/activate
python -m tests.test_agent
```

## Dependencies

- langgraph: For building stateful agents
- langchain: Core LLM framework
- langchain-openai: OpenAI integration
- python-dotenv: Environment variable management

## License

MIT
