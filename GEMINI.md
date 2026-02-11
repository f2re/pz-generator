# Gemini CLI Configuration & Multi-Agent System

This project implements a multi-agent system for generating Jupyter Notebooks from PDF materials using the Gemini CLI.

## 🤖 Agents Overview

The system consists of three specialized agents coordinated by an Orchestrator:

1.  **Analyzer Agent (`.gemini/agents/analyzer.py`):**
    *   **Role:** Educational Material Analyst.
    *   **Task:** Extracts structure, tasks, and theory from PDF files into a structured JSON specification.
    *   **Model:** `gemini-2.0-flash-exp` (Optimized for analysis and large context).

2.  **Programmer Agent (`.gemini/agents/programmer.py`):**
    *   **Role:** Data Science Developer.
    *   **Task:** Converts JSON specifications into fully functional `.ipynb` files with code and explanations.
    *   **Model:** `gemini-2.0-pro-exp` (Optimized for complex coding tasks).

3.  **Tester Agent (`.gemini/agents/tester.py`):**
    *   **Role:** QA Engineer.
    *   **Task:** Validates the generated notebook against the original specification and ensures code quality.
    *   **Model:** `gemini-2.0-flash` (Fast and cost-effective for verification).

## 🛠 Project Structure

- `.gemini/config/`: Contains `agent_config.yaml` for model parameters and `gemini_api.yaml` for API settings.
- `.gemini/prompts/`: System instructions and few-shot examples for each agent.
- `.gemini/workflows/`: YAML definitions of the multi-stage generation process.
- `schemas/`: JSON schemas used for data exchange between agents.

## 🚀 Workflows

### Notebook Generation
The primary workflow follows this path:
`PDF` → `Analyzer` → `JSON Spec` → `Programmer` → `Draft Notebook` → `Tester` → `Review`.
If the `Tester` score is below 70%, the `Programmer` iterates based on the feedback (up to 3 times).

## 🔐 Security & Safety
- **API Keys:** Never commit `gemini_api.yaml` with real keys. Use environment variables in production.
- **Code Execution:** The `Tester Agent` runs generated code. Ensure it executes in a sandboxed environment (e.g., Docker) to prevent system-level impact.
- **Data Privacy:** Be mindful of sensitive data within PDF files processed by the LLM.

## 📈 Monitoring
Logs are stored in the `logs/` directory. Monitor `latency`, `token usage`, and `quality scores` to optimize agent performance.
