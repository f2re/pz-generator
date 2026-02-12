# PZ Generator: PDF to Jupyter Multi-Agent System

An advanced automated pipeline that converts educational PDF lectures and practices into interactive, high-quality Jupyter Notebooks (.ipynb) using a coordinated team of Google Gemini agents.

## 🌟 Features

- **End-to-End Automation:** Complete pipeline from PDF to executed Notebook.
- **Multi-Agent Architecture:** 6 specialized agents working in concert.
- **Pedagogical Quality:** Ensures theory completeness, progressive difficulty, and clear explanations.
- **Automated Testing:** Validates code execution and structural integrity.
- **Self-Correction:** Feedback loops improve content quality automatically.
- **Structured Logging:** Detailed logs in `logs/` for monitoring and debugging.

## 🤖 Agents Overview

The system uses an **Orchestrator** pattern coordinating 5 specialized agents:

1.  **Orchestrator:** Coordinates the entire workflow and manages state.
2.  **Analyzer:** Extracts structured specifications from PDF materials.
3.  **Theory Writer:** Generates methodical theoretical content with examples.
4.  **Programmer:** Creates practical tasks, starter code, and tests.
5.  **Validator:** Ensures alignment between theory and practice.
6.  **Tester:** Validates the final notebook for execution and quality.

See [.gemini/agents/](./.gemini/agents/) for detailed documentation on each agent.

## 🛠 Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd pz-generator
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key:**
   Edit `.gemini/config/gemini_api.yaml` and add your Gemini API key.

## 🚀 Usage

Run the generator. It will automatically detect PDF files in the `pdf/` directory:

```bash
python main.py [optional/path/to/specific.pdf]
```

The generated notebooks will be saved in the `output/` directory, along with reports.

## 🏗 Architecture & Workflow

The generation follows a strict quality-controlled pipeline:

1.  **Analysis:** PDF is analyzed for learning objectives and concepts.
2.  **Theory Generation:** Educational content is created based on the analysis.
3.  **Practice Generation:** Practical tasks are generated to match the theory.
4.  **Validation Loop:** Validator checks if theory covers all practice requirements. If not, Theory Writer adds missing content.
5.  **Testing:** The combined notebook is executed and scored.
6.  **Finalization:** If quality thresholds are met, the result is saved.

## 📁 Project Structure

- `.gemini/agents/`: Agent definitions and configurations.
- `.gemini/config/`: Global system configuration.
- `.gemini/prompts/`: System prompts for LLMs.
- `pdf/`: Input directory for lecture PDFs.
- `output/`: Output directory for generated notebooks.
- `logs/`: Execution logs.

## 🤝 Contributing

Contributions are welcome! Please read the agent documentation before making changes to the agent logic.
