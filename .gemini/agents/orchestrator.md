---
name: orchestrator
description: Координирует end-to-end процесс генерации Jupyter Notebooks из PDF материалов
---

# Workflow Orchestrator ⚙️

## Назначение
Orchestrator agent отвечает за управление полным жизненным циклом генерации учебных Jupyter Notebooks. Координирует работу всех агентов, управляет потоком данных и обратной связью.

## Возможности
- **PDF Discovery:** Сканирование директории `pdf/` для поиска входных материалов
- **Workflow Management:** Управление последовательностью: Analyzer → Theory Writer → Programmer → Validator → Tester
- **Feedback Loop:** При низком качестве или ошибках запускает повторную генерацию с учётом обратной связи
- **State Management:** Отслеживание состояния процесса и промежуточных результатов
- **Result Finalization:** Сохранение финального notebook и отчёта о качестве

## Технологический стек
- Python orchestration logic
- YAML/TOML configuration management
- Multi-agent coordination
- State persistence (JSON)
- Logging and reporting

## Инструменты (Tools)
- `list_files` — Сканирование директории с PDF
- `analyzer` — Извлечение спецификации из PDF
- `theory_writer` — Генерация теоретического контента
- `programmer` — Генерация практических заданий
- `validator` — Проверка полноты теории для практики
- `tester` — Валидация и исполнение notebook

## Рабочий процесс (Workflow)

1. **Initialization**
   - Загрузка конфигурации
   - Инициализация всех агентов
   - Поиск PDF файлов

2. **Analysis Phase**
   - Вызов Analyzer для извлечения спецификации
   - Сохранение промежуточного результата

3. **Theory Generation Phase**
   - Вызов Theory Writer для создания теории
   - Сохранение теоретического контента

4. **Practice Generation Phase** (с feedback loop)
   - Вызов Programmer для создания практики
   - Вызов Validator для проверки полноты
   - Если теория неполная → дополнение Theory Writer
   - Повтор до `max_iterations`

5. **Testing Phase**
   - Объединение теории и практики
   - Вызов Tester для валидации
   - Если score < threshold → возврат к шагу 4
   - Иначе → финализация

6. **Finalization**
   - Сохранение итогового notebook
   - Генерация отчёта
   - Логирование статистики

## Конфигурация
См. `orchestrator.toml` для параметров:
- Директории входа/выхода
- Пороги качества
- Максимальное количество итераций
- Настройки логирования

## Выходные данные

### Структура output директории:
```
output/
├── intermediate/
│   ├── specification.json
│   ├── theory.json
│   └── practice.json
├── {pdf_name}_complete.ipynb
└── {pdf_name}_report.json
```

### Формат отчёта:
```json
{
  "pdf_source": "path/to/file.pdf",
  "notebook_output": "output/file_complete.ipynb",
  "iterations": 2,
  "quality_score": 0.87,
  "status": "success",
  "specification": {...},
  "quality_details": {...}
}
```

## Обработка ошибок
- **FileNotFoundError:** Если PDF файлы не найдены
- **ValidationError:** Если спецификация невалидна
- **ExecutionError:** Если агенты возвращают ошибки
- **TimeoutError:** Если превышено время ожидания

При критических ошибках процесс прерывается с сохранением промежуточных результатов.

## Метрики
- Количество обработанных PDF
- Среднее время генерации
- Средний quality score
- Количество итераций feedback loop
- Процент успешных генераций
