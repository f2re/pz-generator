# PZ Generator: PDF to Jupyter Multi-Agent System

An automated pipeline that converts educational PDF lectures and practices into interactive Jupyter Notebooks (.ipynb) using Google Gemini agents.

## 🌟 Features

- **Automated Extraction:** Analyzes PDF structure to identify tasks and theory.
- **Smart Programming:** Generates idiomatic Python code for Data Science.
- **Self-Correction:** Integrated QA agent that tests notebooks and provides feedback for improvements.
- **Configurable Workflow:** Easily adjust agent behavior via `.gemini/` configuration.

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

Run the generator by providing a path to your PDF file:

```bash
python main.py pdf/your_lecture.pdf
```

The generated notebook will be saved in the `output/` directory.

## 🏗 Architecture

The system uses an **Orchestrator** pattern:
- **Analyzer:** PDF → JSON
- **Programmer:** JSON → .ipynb
- **Tester:** Validation & Review

See [GEMINI.md](./GEMINI.md) for detailed agent documentation.
