import google.generativeai as genai
import nbformat as nbf
import json
import re
from typing import Dict, Any, Optional, List
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
            self.system_instruction = "You are a senior Data Science developer. Create professional, educational Jupyter Notebooks."

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction
        )
    
    def _extract_json(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts JSON cell structure from model response.
        """
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'cells' in data:
                    return data['cells']
            except json.JSONDecodeError:
                pass
        
        # If no markdown block, try parsing whole text
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'cells' in data:
                return data['cells']
        except json.JSONDecodeError:
            pass
            
        return []

    def process(self, json_spec: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate .ipynb file from JSON spec using a holistic approach.
        """
        print(f"[{self.name}] Generating professional notebook...")
        
        prompt = f"Пожалуйста, создай расширенный учебный Notebook на основе следующей спецификации:\n\n{json.dumps(json_spec, indent=2, ensure_ascii=False)}"
        
        if feedback:
            prompt += f"\n\n### Обратная связь для исправления:\n{json.dumps(feedback, indent=2, ensure_ascii=False)}"
            prompt += "\n\nПожалуйста, учти эту обратную связь и исправь ошибки в новой версии Notebook."

        try:
            response = self.model.generate_content(prompt)
            cells_data = self._extract_json(response.text)
            
            if not cells_data:
                print(f"[{self.name}] Warning: Failed to parse structured JSON. Falling back to raw text processing.")
                # Basic fallback if JSON extraction fails - we could implement a more robust fallback here
                return self._fallback_generation(json_spec, response.text)

            nb = nbf.v4.new_notebook()
            
            for cell in cells_data:
                cell_type = cell.get('type', 'markdown')
                content = cell.get('content', '')
                
                # Content can be a list or a string
                if isinstance(content, list):
                    content = "\n".join(content)
                
                if cell_type == 'code':
                    nb.cells.append(nbf.v4.new_code_cell(content))
                else:
                    nb.cells.append(nbf.v4.new_markdown_cell(content))

            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            # Use practice number for filename if available
            p_num = json_spec.get('practices', [{}])[0].get('number', 'gen')
            output_path = output_dir / f"practice_{p_num}_advanced.ipynb"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
                
            print(f"[{self.name}] Success! Notebook saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            return "output/failed_generation.ipynb"

    def _fallback_generation(self, json_spec: Dict[str, Any], raw_text: str) -> str:
        """
        Simple fallback to at least save the model's text if JSON parsing fails.
        """
        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_markdown_cell("# Generated Notebook (Recovery Mode)"))
        nb.cells.append(nbf.v4.new_markdown_cell("Parsing of structured JSON failed. Here is the raw output from the model:"))
        
        # Split by what looks like code blocks
        parts = re.split(r'```python|```', raw_text)
        for i, part in enumerate(parts):
            if not part.strip(): continue
            if i % 2 == 1: # This was inside ```python ... ```
                nb.cells.append(nbf.v4.new_code_cell(part.strip()))
            else:
                nb.cells.append(nbf.v4.new_markdown_cell(part.strip()))
                
        output_path = Path("output/recovered_notebook.ipynb")
        with open(output_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        return str(output_path)
