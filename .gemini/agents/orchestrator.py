import yaml
from pathlib import Path
from typing import Dict, Any, List

class Orchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Инициализация всех агентов из конфигурации"""
        from .analyzer import AnalyzerAgent
        from .programmer import ProgrammerAgent
        from .tester import TesterAgent
        
        self.agents['analyzer'] = AnalyzerAgent(
            self.config['agents']['analyzer']
        )
        self.agents['programmer'] = ProgrammerAgent(
            self.config['agents']['programmer']
        )
        self.agents['tester'] = TesterAgent(
            self.config['agents']['tester']
        )
    
    def run(self, pdf_path: str, workflow_config_path: str):
        """Запуск workflow из .gemini/workflows/"""
        workflow = self._load_workflow(workflow_config_path)
        
        state = {
            'pdf_path': pdf_path,
            'iteration': 0
        }
        
        # Stage 1: Analyze
        state['json_spec'] = self.agents['analyzer'].process(
            state['pdf_path']
        )
        
        # Stage 2-3: Program & Test Loop
        max_iter = self.config['orchestrator'].get('max_iterations', 3)
        min_score = self.config['orchestrator'].get('min_quality_score', 70)
        
        while state['iteration'] < max_iter:
            # Program
            state['notebook'] = self.agents['programmer'].process(
                state['json_spec'],
                feedback=state.get('feedback')
            )
            
            # Test
            review = self.agents['tester'].process(
                state['notebook'],
                state['json_spec']
            )
            
            if review.get('score', 0) >= min_score:
                return self._success_result(state, review)
            
            state['feedback'] = review
            state['iteration'] += 1
        
        # Escalate after max iterations
        return self._escalate_result(state)
    
    def _load_workflow(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _success_result(self, state: Dict[str, Any], review: Dict[str, Any]):
        return type('Result', (), {
            'status': 'success',
            'output_path': state['notebook'],
            'score': review['score'],
            'error': None,
            'needs_human_review': False
        })

    def _escalate_result(self, state: Dict[str, Any]):
        return type('Result', (), {
            'status': 'failed',
            'output_path': state.get('notebook'),
            'score': state.get('feedback', {}).get('score', 0),
            'error': 'Max iterations reached without passing score',
            'needs_human_review': True
        })
