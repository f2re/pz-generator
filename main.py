import os
import yaml
import sys
from pathlib import Path

# Add .gemini directory to path to allow importing from agents
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / ".gemini"))

from agents.orchestrator import OrchestratorAgent

def load_agent_config():

def load_agent_config():
    """Загрузка конфигурации из .gemini/"""
    config_path = current_dir / ".gemini" / "config" / "agent_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main(pdf_path: Optional[str] = None):
    # Загрузить конфигурацию агентов
    config = load_agent_config()
    
    # Инициализировать оркестратор
    orchestrator = OrchestratorAgent(config)
    
    # Запустить workflow
    # Orchestrator сам найдет PDF в папке pdf/, если pdf_path не передан
    result = orchestrator.run(pdf_path=pdf_path)
    
    if result.status == "success":
        print(f"✓ Notebook generated: {result.output_path}")
        print(f"  Quality score: {result.score}/100")
    else:
        print(f"✗ Generation failed: {result.error}")
        if result.needs_human_review:
            print(f"  → Escalated for human review")

if __name__ == "__main__":
    from typing import Optional
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(path_arg)
