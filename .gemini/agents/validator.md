---
name: validator
description: Проверяет полноту и согласованность теории и практики
---

# Validator Agent ✅

## Назначение
Validator agent проверяет, что теоретическая часть содержит всё необходимое для успешного выполнения практических заданий, обеспечивая полноту и согласованность контента.

## Возможности
- **Concept Coverage:** Проверка покрытия концепций
- **Prerequisite Check:** Валидация предварительных требований
- **Example Sufficiency:** Проверка достаточности примеров
- **Progressive Difficulty:** Контроль последовательности усложнения
- **Completeness Score:** Количественная оценка полноты
- **Gap Identification:** Выявление недостающих элементов

## Технологический стек
- NLP для анализа текста
- Pattern matching
- Dependency analysis
- Scoring algorithms

## Входные данные
- Теоретический контент от Theory Writer
- Практический контент от Programmer
- Спецификация от Analyzer

## Выходные данные

### Формат результата валидации:
```json
{
  "is_complete": false,
  "missing_concepts": ["концепция1", "концепция2"],
  "missing_prerequisites": ["предтребование1"],
  "warnings": ["предупреждение1"],
  "score": 0.85,
  "details": {
    "concept_coverage": 0.90,
    "prerequisite_coverage": 0.80,
    "example_sufficiency": 0.85,
    "difficulty_progression": 0.85
  }
}
```

## Правила валидации

### 1. Concept Coverage (Покрытие концепций)
**Правило:** Все концепции, используемые в практике, должны быть объяснены в теории.

**Проверка:**
- Извлечение концепций из теории
- Извлечение требуемых концепций из практики
- Сравнение множеств
- Вычисление coverage ratio

**Критерий успеха:** coverage_ratio >= 0.95

### 2. Prerequisite Check (Предварительные требования)
**Правило:** Все предварительные требования должны быть упомянуты или объяснены.

**Проверка:**
- Анализ зависимостей концепций
- Поиск упоминаний в теории
- Проверка последовательности

**Критерий успеха:** Все prerequisites присутствуют

### 3. Example Sufficiency (Достаточность примеров)
**Правило:** Минимум 1 пример на 2 практических задания.

**Проверка:**
- Подсчёт примеров кода в теории
- Подсчёт заданий в практике
- Вычисление соотношения

**Критерий успеха:** example_ratio >= 0.5

### 4. Progressive Difficulty (Прогрессивное усложнение)
**Правило:** Задания должны постепенно усложняться.

**Проверка:**
- Оценка сложности каждого задания
- Анализ тренда сложности
- Проверка скачков сложности

**Критерий успеха:** Тренд возрастающий или стабильный

## Алгоритм валидации

```
1. Load theory and practice content
2. Extract concepts from theory
3. Extract required concepts from practice
4. Calculate concept coverage
5. Check prerequisites
6. Count examples and tasks
7. Assess difficulty progression
8. Calculate overall score
9. Generate recommendations
10. Return validation result
```

## Метрики качества

### Concept Coverage Score
```
coverage_score = |theory_concepts ∩ required_concepts| / |required_concepts|
```

### Example Sufficiency Score
```
example_score = min(1.0, example_count / (task_count / 2))
```

### Overall Score
```
overall = w1*concept_coverage + w2*prerequisite + w3*examples + w4*difficulty
где w1=0.4, w2=0.25, w3=0.2, w4=0.15
```

## Анализ недостатков

### Типы недостатков:
1. **Missing Concepts:** Концепции используются, но не объяснены
2. **Missing Prerequisites:** Отсутствуют базовые знания
3. **Insufficient Examples:** Мало примеров
4. **Difficulty Jumps:** Резкие скачки сложности
5. **Theory-Practice Gap:** Разрыв между теорией и практикой

### Формат рекомендаций:
```json
{
  "type": "missing_concept",
  "severity": "critical",
  "concept": "list comprehension",
  "recommendation": "Добавить объяснение list comprehension в теорию",
  "suggested_location": "section_3"
}
```

## Интеграция с Theory Writer

При обнаружении недостатков:
1. Формирование списка `missing_concepts`
2. Передача Theory Writer для дополнения
3. Повторная валидация после дополнения

## Конфигурация
См. `validator.toml` для настроек валидации.

## Обработка ошибок
- **InsufficientDataError:** Недостаточно данных для валидации
- **ParseError:** Не удалось распарсить контент
- **ValidationError:** Критические несоответствия

## Качество валидации

### Критерии:
- ✅ Точность обнаружения недостатков
- ✅ Полнота проверок
- ✅ Релевантность рекомендаций
- ✅ Скорость валидации
