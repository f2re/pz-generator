import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

class OrchestratorAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "Orchestrator"
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents from configuration"""
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
    
    def find_pdf(self) -> Optional[str]:
        """Find first PDF file in the pdf/ directory"""
        pdf_dir = Path("pdf")
        if not pdf_dir.exists():
            pdf_dir.mkdir(exist_ok=True)
            return None
            
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            return None
            
        return str(pdf_files[0])

    def run(self, pdf_path: Optional[str] = None, workflow_config_path: Optional[str] = None):
        """Execute the full generation workflow"""
        print(f"[{self.name}] Starting workflow...")
        
        # Discover PDF if not provided
        if pdf_path is None:
            pdf_path = self.find_pdf()
            if not pdf_path:
                return type('Result', (), {
                    'status': 'failed',
                    'output_path': None,
                    'score': 0,
                    'error': "No PDF files found in 'pdf/' directory",
                    'needs_human_review': False
                })
            print(f"[{self.name}] Discovered PDF: {pdf_path}")

        state = {
            'pdf_path': pdf_path,
            'iteration': 0,
            'feedback': None
        }
        
        # Stage 1: Analyze
        try:
            state['json_spec'] = self.agents['analyzer'].process(
                state['pdf_path']
            )
        except Exception as e:
            return self._error_result(f"Analyzer failed: {e}")
        
        # Stage 2-3: Program & Test Loop
        max_iter = self.config['orchestrator'].get('max_iterations', 3)
        min_score = self.config['orchestrator'].get('min_quality_score', 70)
        
        while state['iteration'] < max_iter:
            print(f"[{self.name}] Iteration {state['iteration'] + 1}/{max_iter}")
            
            # Program
            try:
                state['notebook'] = self.agents['programmer'].process(
                    state['json_spec'],
                    feedback=state.get('feedback')
                )
            except Exception as e:
                return self._error_result(f"Programmer failed at iter {state['iteration']}: {e}")
            
            # Test
            try:
                review = self.agents['tester'].process(
                    state['notebook'],
                    state['json_spec']
                )
            except Exception as e:
                return self._error_result(f"Tester failed at iter {state['iteration']}: {e}")
            
            score = review.get('score', 0)
            print(f"[{self.name}] Quality score: {score}")
            
            if score >= min_score:
                return self._success_result(state, review)
            
            state['feedback'] = review
            state['iteration'] += 1
        
        # Escalate after max iterations
        return self._escalate_result(state)
    
    def _error_result(self, message: str):
        print(f"[{self.name}] Error: {message}")
        return type('Result', (), {
            'status': 'failed',
            'output_path': None,
            'score': 0,
            'error': message,
            'needs_human_review': True
        })

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
