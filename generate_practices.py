
import sys
import yaml
import json
from pathlib import Path
from typing import List

# Add .gemini to path
sys.path.append(str(Path(__file__).parent / ".gemini"))

from agents.orchestrator import OrchestratorAgent

def load_config():
    with open(".gemini/config/agent_config.yaml", 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_specific_practices(nums: List[int]):
    config = load_config()
    orchestrator = OrchestratorAgent(config)
    
    # Load spec
    with open("output/practice_spec.json", 'r', encoding='utf-8') as f:
        full_spec = json.load(f)
    
    practices = full_spec.get('practices', [])
    for p_num in nums:
        practice = next((p for p in practices if p['number'] == p_num), None)
        if not practice:
            print(f"Practice {p_num} not found in spec.")
            continue
            
        print(f"\n>>> Starting Generation for Practice №{p_num} <<<")
        practice_spec = {"practices": [practice]}
        result = orchestrator._process_single_practice(practice_spec)
        
        if result.status == "success":
            print(f"✓ Practice {p_num} completed: {result.output_path}")
        else:
            print(f"✗ Practice {p_num} failed: {result.error}")

if __name__ == "__main__":
    to_generate = [3, 4, 5, 6]
    if len(sys.argv) > 1:
        to_generate = [int(x) for x in sys.argv[1:]]
    
    generate_specific_practices(to_generate)
