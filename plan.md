
```markdown
# План разработки многоагентной системы генерации Jupyter практикумов

## 1. Архитектура системы

### 1.1 Общая структура
```

Orchestrator (Coordinator Agent)
├── Analyzer Agent (PDF → JSON)
├── Programmer Agent (JSON → .ipynb)
└── Tester Agent (validation \& review)

```

### 1.2 Технологический стек
- **LLM**: Google Gemini 2.0/3.0 Pro (расширенный context window, multimodal)
- **Orchestration**: LangGraph или custom state machine
- **PDF Processing**: Gemini native PDF + PyMuPDF для fallback
- **Notebook**: nbformat для создания .ipynb
- **Validation**: nbconvert, papermill для тестирования
- **Storage**: PostgreSQL для истории задач и версионирования

---

## 2. Агент #1: Analyzer (Анализатор)

### 2.1 Роль и ответственность
- Принимает PDF файл с лекцией/практикумом
- Извлекает структуру: номера практик, задачи, теоретические основы
- Формирует JSON спецификацию для Programmer Agent

### 2.2 System Instructions
```

Ты — аналитик образовательных материалов. Твоя задача:

1. Прочитать PDF и идентифицировать все практические задания
2. Для каждой практики извлечь:
    - Номер практики
    - Название/тему
    - Теоретический контекст
    - Входные данные
    - Требуемые шаги решения
    - Ожидаемый результат
3. Выдать структурированный JSON

Формат выхода:
{
"practices": [
{
"number": 1,
"title": "Линейная регрессия",
"theory": "краткое описание теории",
"tasks": [
{
"task_id": "1.1",
"description": "Загрузить датасет",
"inputs": ["data.csv"],
"steps": ["pd.read_csv()", "проверить shape"],
"expected_output": "DataFrame 100x5"
}
],
"libraries": ["pandas", "numpy", "sklearn"],
"difficulty": "intermediate"
}
]
}

```

### 2.3 Tools & Function Calls
- `extract_pdf_structure()`: Gemini native PDF processing
- `identify_code_blocks()`: Обнаружение примеров кода в PDF
- `validate_json_schema()`: Проверка корректности JSON

### 2.4 Self-Correction
- Если JSON невалиден → retry с указанием ошибки
- Если пропущены задачи → re-scan PDF с фокусом на пропущенные секции

---

## 3. Агент #2: Programmer (Программист)

### 3.1 Роль и ответственность
- Получает JSON спецификацию от Analyzer
- Генерирует полностью рабочий .ipynb файл
- Добавляет комментарии, docstrings, markdown ячейки с объяснениями

### 3.2 System Instructions
```

Ты — опытный Python разработчик для Data Science образования. Твоя задача:

1. Прочитать JSON спецификацию практики
2. Создать Jupyter Notebook с:
    - Markdown заголовками и теорией
    - Импортами библиотек
    - Комментированным кодом для каждого шага
    - Placeholder для данных пользователя
    - Примерами выхода
3. Код ДОЛЖЕН быть:
    - Синтаксически корректным
    - Готовым к запуску (с example data)
    - Педагогически понятным (не overcomplicated)

Структура notebook:

1. Заголовок практики (Markdown)
2. Теория (Markdown)
3. Импорты (Code)
4. Шаг 1 (Markdown + Code)
...
N. Выводы (Markdown)
```

### 3.3 Tools & Function Calls
- `generate_notebook_cell()`: Создание ячейки ipynb
- `add_markdown_explanation()`: Генерация пояснительного текста
- `validate_syntax()`: Проверка Python кода через ast.parse
- `suggest_visualizations()`: Предложение графиков для результатов

### 3.4 Context Management
- Использовать **just-in-time loading** для больших примеров данных
- Кэшировать часто используемые паттерны кода (templates)

---

## 4. Агент #3: Tester (Тестировщик)

### 4.1 Роль и ответственность
- Получает .ipynb от Programmer
- Запускает notebook в изолированной среде
- Валидирует соответствие исходной задаче
- Возвращает JSON с ревизией

### 3.2 System Instructions
```

Ты — QA инженер для образовательных Jupyter Notebooks. Твоя задача:

1. Запустить notebook в чистом окружении
2. Проверить:
    - Все ячейки выполняются без ошибок
    - Выход соответствует ожиданиям из JSON спецификации
    - Комментарии понятны и достаточны
    - Нет hardcoded путей/данных (кроме примеров)
3. Сравнить с исходной спецификацией:
    - Все задачи покрыты
    - Логика соответствует теории
    - Уровень сложности адекватен

Формат выхода:
{
"status": "pass/fail",
"errors": [
{"cell": 5, "error": "NameError: numpy not imported"}
],
"missing_tasks": ["task_id: 2.3"],
"logic_mismatches": ["Используется mean вместо median"],
"suggestions": ["Добавить проверку на NaN"],
"score": 85
}

```

### 4.3 Tools & Function Calls
- `execute_notebook()`: Запуск через papermill/nbconvert
- `compare_with_spec()`: Сопоставление с JSON от Analyzer
- `static_code_analysis()`: pylint/flake8 для качества кода
- `check_pedagogical_quality()`: Оценка понятности для студентов

### 4.4 Retry Logic
- Если score < 70 → отправить обратно Programmer с детальным feedback
- Максимум 3 итерации, затем эскалация Orchestrator

---

## 5. Orchestrator (Координатор)

### 5.1 Роль
- Управляет workflow между агентами
- Обрабатывает ошибки и retry логику
- Логирует все этапы для аудита
- Решает конфликты между агентами

### 5.2 State Machine (LangGraph)
```

START
↓
[Analyzer] → JSON spec
↓
[Programmer] → .ipynb draft
↓
[Tester] → review JSON
↓
├─ PASS → END (сохранить .ipynb)
└─ FAIL → [Programmer] (с feedback, max 3 iterations)
↓
└─ Still FAIL → ESCALATION (human review)

```

### 5.3 Context Sharing
```python
class SharedState:
    pdf_path: str
    json_spec: dict
    notebook_path: str
    test_results: dict
    iteration_count: int
    messages: List[AgentMessage]
```


### 5.4 Function Declarations

```python
tools = [
    {
        "name": "route_to_agent",
        "description": "Направить задачу нужному агенту",
        "parameters": {
            "agent": {"type": "string", "enum": ["analyzer", "programmer", "tester"]},
            "payload": {"type": "object"}
        }
    },
    {
        "name": "save_artifact",
        "description": "Сохранить промежуточный результат",
        "parameters": {
            "artifact_type": {"type": "string"},
            "content": {"type": "string"}
        }
    }
]
```


---

## 6. Реализация: Пошаговый план

### Phase 1: Прототип (1-2 недели)

- [ ] Настроить Gemini API с function calling
- [ ] Создать базовую структуру Analyzer агента
- [ ] Тест на простом PDF (1 практика)
- [ ] Валидация JSON выхода


### Phase 2: Full Pipeline (2-3 недели)

- [ ] Реализовать Programmer агента
- [ ] Генерация .ipynb с nbformat
- [ ] Интеграция Analyzer → Programmer
- [ ] Ручное тестирование сгенерированных notebooks


### Phase 3: Testing \& Iteration (2 недели)

- [ ] Создать Tester агента с papermill
- [ ] Реализовать retry logic
- [ ] Интеграция всех трех агентов через LangGraph
- [ ] E2E тестирование на 5+ различных PDF


### Phase 4: Orchestration \& Production (1-2 недели)

- [ ] Финализировать Orchestrator
- [ ] Добавить логирование (PostgreSQL + structured logs)
- [ ] Context caching для оптимизации токенов
- [ ] Мониторинг метрик (latency, cost, success rate)
- [ ] Документация и примеры использования



---

## 7. Best Practices \& Considerations

### 7.1 Prompt Engineering

- Использовать **few-shot examples** для сложных преобразований
- Явно указывать формат выхода (JSON schema)
- Добавлять guardrails: "Never generate code with hardcoded credentials"


### 7.2 Error Handling

- Каждый агент должен возвращать статус: success/partial/fail
- Graceful degradation: если Tester не может запустить → ручная проверка
- Сохранять все промежуточные результаты для debugging


```
project/
├── .gemini/                    # Конфигурация агентов (аналог .github)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── analyzer.py         # Analyzer Agent
│   │   ├── programmer.py       # Programmer Agent
│   │   ├── tester.py          # Tester Agent
│   │   └── orchestrator.py    # Main Orchestrator
│   ├── prompts/
│   │   ├── analyzer_prompt.txt
│   │   ├── programmer_prompt.txt
│   │   └── tester_prompt.txt
│   ├── config/
│   │   ├── agent_config.yaml   # Конфигурация каждого агента
│   │   └── gemini_api.yaml     # API keys, model settings
│   └── workflows/
│       ├── notebook_generation.yaml  # Описание workflow
│       └── state_machine.json        # LangGraph граф
├── schemas/
│   ├── practice_spec.json      # JSON schema для спецификации
│   └── review_result.json      # JSON schema для ревизии
├── tools/
│   ├── pdf_parser.py           # Gemini PDF + fallback
│   ├── notebook_generator.py   # nbformat utilities
│   └── notebook_runner.py      # papermill executor
├── tests/
│   ├── test_analyzer.py
│   ├── test_programmer.py
│   └── test_integration.py
├── examples/
│   ├── sample_lecture.pdf
│   └── expected_output.ipynb
├── logs/
├── output/                     # Сгенерированные notebooks
├── requirements.txt
├── main.py                     # Entry point
└── README.md
```


## Конфигурация агентов в `.gemini/`

### `.gemini/config/agent_config.yaml`

```yaml
agents:
  analyzer:
    name: "PDF Analyzer Agent"
    model: "gemini-2.0-flash-exp"
    temperature: 0.1
    max_tokens: 8192
    system_instruction_path: ".gemini/prompts/analyzer_prompt.txt"
    tools:
      - extract_pdf_structure
      - identify_code_blocks
      - validate_json_schema
    retry_limit: 3
    
  programmer:
    name: "Notebook Programmer Agent"
    model: "gemini-2.0-pro-exp"
    temperature: 0.3
    max_tokens: 16384
    system_instruction_path: ".gemini/prompts/programmer_prompt.txt"
    tools:
      - generate_notebook_cell
      - add_markdown_explanation
      - validate_syntax
      - suggest_visualizations
    context_caching: true
    retry_limit: 3
    
  tester:
    name: "Notebook Tester Agent"
    model: "gemini-2.0-flash"  # Cheaper для testing
    temperature: 0.0
    max_tokens: 8192
    system_instruction_path: ".gemini/prompts/tester_prompt.txt"
    tools:
      - execute_notebook
      - compare_with_spec
      - static_code_analysis
      - check_pedagogical_quality
    execution_timeout: 300  # 5 minutes
    retry_limit: 3

orchestrator:
  max_iterations: 3
  min_quality_score: 70
  enable_human_review: true
  human_review_threshold: 50
```


### `.gemini/workflows/notebook_generation.yaml`

```yaml
workflow:
  name: "PDF to Jupyter Notebook Generation"
  version: "1.0.0"
  
  stages:
    - name: "analyze"
      agent: "analyzer"
      input: "pdf_file"
      output: "json_spec"
      validation:
        - type: "json_schema"
          schema: "schemas/practice_spec.json"
      on_failure: "retry_or_escalate"
      
    - name: "program"
      agent: "programmer"
      input: "json_spec"
      output: "notebook_file"
      validation:
        - type: "file_exists"
        - type: "ipynb_format"
      on_failure: "retry_with_feedback"
      
    - name: "test"
      agent: "tester"
      input: ["notebook_file", "json_spec"]
      output: "review_json"
      validation:
        - type: "score_threshold"
          min_score: 70
      on_failure: "feedback_loop"
      
  feedback_loop:
    max_iterations: 3
    stages: ["program", "test"]
    escalate_after: 3
```


## Пример кода для загрузки агентов

### `main.py`

```python
import os
import yaml
from pathlib import Path
from .gemini.agents.orchestrator import Orchestrator

def load_agent_config():
    """Загрузка конфигурации из .gemini/"""
    config_path = Path(".gemini/config/agent_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(pdf_path: str):
    # Загрузить конфигурацию агентов
    config = load_agent_config()
    
    # Инициализировать оркестратор
    orchestrator = Orchestrator(config)
    
    # Запустить workflow
    result = orchestrator.run(
        pdf_path=pdf_path,
        workflow_config=".gemini/workflows/notebook_generation.yaml"
    )
    
    if result.status == "success":
        print(f"✓ Notebook generated: {result.output_path}")
        print(f"  Quality score: {result.score}/100")
    else:
        print(f"✗ Generation failed: {result.error}")
        if result.needs_human_review:
            print(f"  → Escalated for human review")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)
    
    main(sys.argv[^1])
```


### `.gemini/agents/orchestrator.py`

```python
import yaml
from pathlib import Path
from typing import Dict, Any
import google.generativeai as genai

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
    
    def run(self, pdf_path: str, workflow_config: str):
        """Запуск workflow из .gemini/workflows/"""
        workflow = self._load_workflow(workflow_config)
        
        state = {
            'pdf_path': pdf_path,
            'iteration': 0
        }
        
        # Stage 1: Analyze
        state['json_spec'] = self.agents['analyzer'].process(
            state['pdf_path']
        )
        
        # Stage 2-3: Program & Test Loop
        max_iter = self.config['orchestrator']['max_iterations']
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
            
            if review['score'] >= 70:
                return self._success_result(state, review)
            
            state['feedback'] = review
            state['iteration'] += 1
        
        # Escalate after max iterations
        return self._escalate_result(state)
    
    def _load_workflow(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
```


