import google.generativeai as genai
import nbformat as nbf
import json
from typing import Dict, Any, Optional
from pathlib import Path

class ProgrammerAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'Programmer')
        self.model_name = config.get('model', 'gemini-2.0-pro-exp')
        
        prompt_path = Path(config.get('system_instruction_path', '.gemini/prompts/programmer_prompt.txt'))
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_instruction = f.read()
        else:
            self.system_instruction = "You are a Jupyter Notebook programmer."

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction
        )
    
    def process(self, json_spec: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate .ipynb file from JSON spec.
        """
        print(f"[{self.name}] Generating notebook from spec...")
        
        prompt = f"JSON Specification: {json.dumps(json_spec, indent=2, ensure_ascii=False)}"
        if feedback:
            prompt += f"\n\nPrevious Feedback to address: {json.dumps(feedback, indent=2, ensure_ascii=False)}"
        
        try:
            response = self.model.generate_content(prompt)
            # The agent is expected to return code or a structured response.
            # For simplicity, we'll ask it to provide the content for the notebook.
            # Alternatively, we could have it generate the whole .ipynb string.
            
            # Let's assume it returns a list of cells in a specific JSON format or we parse it.
            # For this implementation, we'll create a basic notebook based on the spec
            # and use the model to fill in the code blocks if needed.
            
            nb = nbf.v4.new_notebook()
            
            for practice in json_spec.get('practices', []):
                nb.cells.append(nbf.v4.new_markdown_cell(f"# Practice {practice.get('number')}: {practice.get('title')}"))
                nb.cells.append(nbf.v4.new_markdown_cell(f"## Theory\n{practice.get('theory')}"))
                
                if practice.get('libraries'):
                    libs = ", ".join(practice.get('libraries'))
                    nb.cells.append(nbf.v4.new_code_cell(f"# Imports\nimport " + "\nimport ".join(practice.get('libraries'))))
                
                for task in practice.get('tasks', []):
                    nb.cells.append(nbf.v4.new_markdown_cell(f"### Task {task.get('task_id')}\n{task.get('description')}"))
                    # Ask model for specific code for this task
                    task_prompt = f"Generate Python code for the following task from practice '{practice.get('title')}':\n{json.dumps(task, indent=2, ensure_ascii=False)}"
                    task_code_response = self.model.generate_content(task_prompt)
                    
                    code = task_code_response.text
                    # Clean markdown code blocks if present
                    if "```python" in code:
                        code = code.split("```python")[1].split("```")[0].strip()
                    elif "```" in code:
                        code = code.split("```")[1].split("```")[0].strip()
                        
                    nb.cells.append(nbf.v4.new_code_cell(code))

            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "generated_notebook.ipynb"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
                
            return str(output_path)
            
        except Exception as e:
            print(f"[{self.name}] Error during notebook generation: {e}")
            return "output/failed_notebook.ipynb"
