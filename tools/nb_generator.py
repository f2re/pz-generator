import json
import os

def create_notebook(practice):
    cells = []
    
    # Title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Практическое занятие №" + str(practice['number']) + "\n",
            "## " + practice['title'] + "\n"
        ]
    })
    
    # Theory
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Теоретические сведения\n",
            practice['theory'] + "\n"
        ]
    })
    
    # Imports
    refined_imports = []
    for lib in practice['libraries']:
        if lib == 'pandas':
            refined_imports.append("import pandas as pd")
        elif lib == 'numpy':
            refined_imports.append("import numpy as np")
        elif lib == 'matplotlib':
            refined_imports.append("import matplotlib.pyplot as plt")
        elif lib == 'seaborn':
            refined_imports.append("import seaborn as sns")
        elif lib == 'sklearn':
            refined_imports.append("from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.metrics import classification_report, mean_squared_error, r2_score")
        elif lib == 'keras':
            refined_imports.append("from tensorflow import keras\nfrom tensorflow.keras import layers")
        elif lib == 'tensorflow':
            refined_imports.append("import tensorflow as tf")
        elif lib == 'clickhouse_driver':
            refined_imports.append("from clickhouse_driver import Client")
        else:
            refined_imports.append("import " + lib)
            
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Импорт необходимых библиотек\n",
            "\n".join(refined_imports) + "\n"
        ]
    })
    
    # Tasks
    for task in practice['tasks']:
        task_desc = [
            "### Задание " + task['task_id'] + "\n",
            "**Описание:** " + task['description'] + "\n\n",
            "**Входные данные:** " + ", ".join(task['inputs']) + "\n\n",
            "**Шаги реализации:**\n"
        ]
        for i, step in enumerate(task['steps']):
            task_desc.append(str(i+1) + ". " + step + "\n")
        
        task_desc.append("\n**Ожидаемый результат:** " + task['expected_output'] + "\n")
        
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": task_desc
        })
        
        # Code placeholder for the task
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Решение задания " + task['task_id'] + "\n",
                "# TODO: Ваш код здесь\n"
            ]
        })
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def main():
    spec_path = '/home/YaremenkoIA/pz-generator/output/practice_spec.json'
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    
    output_dir = '/home/YaremenkoIA/pz-generator/output'
    for practice in spec['practices']:
        nb = create_notebook(practice)
        filename = "practice_" + str(practice['number']) + ".ipynb"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("Generated " + filename)

if __name__ == "__main__":
    main()
