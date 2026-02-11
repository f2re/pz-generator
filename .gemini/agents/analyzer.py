import google.generativeai as genai
from typing import Dict, Any
from pathlib import Path

class AnalyzerAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'Analyzer')
        self.model_name = config.get('model', 'gemini-2.0-flash-exp')
        
        # Load system instruction
        prompt_path = Path(config.get('system_instruction_path', '.gemini/prompts/analyzer_prompt.txt'))
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.system_instruction = f.read()
        else:
            self.system_instruction = "You are a PDF analyzer."

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_instruction
        )
    
    def process(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process PDF and extract practice structure.
        """
        print(f"[{self.name}] Analyzing PDF: {pdf_path}")
        
        # In a real implementation, we would upload the file to Gemini
        # or read it and send as parts. For now, let's assume multimodal support.
        # Note: genai.upload_file is the recommended way for large PDFs.
        
        # For simplicity in this fix, we'll use a placeholder logic that 
        # simulates a call, as we might not have a real PDF yet.
        # But let's write the code as if it's functional.
        
        try:
            # Upload the file
            sample_pdf = genai.upload_file(path=pdf_path, display_name="Input Material")
            
            response = self.model.generate_content([
                sample_pdf,
                "Please extract the practices according to your system instructions."
            ])
            
            # Extract JSON from response (handling potential markdown formatting)
            import json
            import re
            
            content = response.text
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            return json.loads(content)
            
        except Exception as e:
            print(f"[{self.name}] Error during analysis: {e}")
            # Fallback/Dummy for testing if API call fails
            return {
                "practices": [
                    {
                        "number": 1,
                        "title": f"Analysis of {Path(pdf_path).name}",
                        "theory": "Failed to extract theory automatically.",
                        "tasks": [{"task_id": "1", "description": "Manual review required"}],
                        "libraries": [],
                        "difficulty": "beginner"
                    }
                ]
            }
