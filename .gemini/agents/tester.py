import google.generativeai as genai
import papermill as pm
import json
from typing import Dict, Any
from pathlib import Path

class TesterAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'Tester')
        self.model_name = config.get('model', 'gemini-2.0-flash')
        
        prompt_path = Path(config.get('system_instruction_path', '.gemini/prompts/tester_prompt.txt'))
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_instruction = f.read()
        else:
            self.system_instruction = "You are a Jupyter Notebook tester."

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction
        )
    
    def process(self, notebook_path: str, json_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate notebook against spec and execute it.
        """
        print(f"[{self.name}] Testing notebook: {notebook_path}")
        
        execution_success = True
        execution_error = None
        
        try:
            # Execute the notebook
            temp_output = "logs/test_execution.ipynb"
            Path("logs").mkdir(exist_ok=True)
            pm.execute_notebook(
                notebook_path,
                temp_output,
                kernel_name='python3'
            )
        except Exception as e:
            execution_success = False
            execution_error = str(e)
            print(f"[{self.name}] Execution failed: {e}")

        # Now ask Gemini to review the notebook (or its content)
        try:
            # For simplicity, we'll read the notebook content
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb_content = f.read()
                
            review_prompt = f"""
            Compare this notebook with the original specification.
            
            Original Specification:
            {json.dumps(json_spec, indent=2, ensure_ascii=False)}
            
            Execution Status: {'Success' if execution_success else 'Failed'}
            Execution Error: {execution_error}
            
            Notebook Content:
            {nb_content[:10000]} # Truncated for token limits
            """
            
            response = self.model.generate_content(review_prompt)
            
            content = response.text
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            review_result = json.loads(content)
            return review_result
            
        except Exception as e:
            print(f"[{self.name}] Error during testing review: {e}")
            return {
                "status": "fail" if not execution_success else "pass",
                "score": 50 if not execution_success else 70,
                "errors": [execution_error] if execution_error else [],
                "suggestions": ["Failed to perform full AI review."]
            }
