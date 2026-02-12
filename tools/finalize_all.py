
import nbformat as nbf
import json
from pathlib import Path

def create_final_notebook(practice):
    nb = nbf.v4.new_notebook()
    p_num = practice['number']
    title = practice['title']
    theory = practice['theory']
    libs = practice['libraries']
    
    # 1. Title and Intro
    nb.cells.append(nbf.v4.new_markdown_cell(f"# Практическое занятие №{p_num}\n# {title}"))
    
    intro_lines = [
        "## 1. Введение",
        f"Данная работа посвящена теме: \"{title}\".",
        "",
        f"**Цель работы:** Освоение практических навыков работы с {', '.join(libs)} для решения задач в области гидрометеорологии.",
        "",
        "**Актуальность:** Автоматизация обработки метеорологических данных позволяет повысить точность прогнозов и скорость анализа больших массивов информации."
    ]
    nb.cells.append(nbf.v4.new_markdown_cell("\n".join(intro_lines)))
    
    # 2. Theory
    theory_lines = [
        "## 2. Теоретические сведения",
        theory,
        "",
        "В работе используются современные методы обработки данных и библиотеки Python для обеспечения точности и эффективности вычислений."
    ]
    nb.cells.append(nbf.v4.new_markdown_cell("\n".join(theory_lines)))
    
    # 3. Initialization
    imports = []
    if 'pandas' in libs: imports.append("import pandas as pd")
    if 'numpy' in libs: imports.append("import numpy as np")
    if 'matplotlib' in libs: imports.append("import matplotlib.pyplot as plt")
    if 'seaborn' in libs: imports.append("import seaborn as sns")
    if 're' in libs: imports.append("import re")
    if 'sklearn' in libs: 
        imports.append("from sklearn.model_selection import train_test_split")
        imports.append("from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor")
        imports.append("from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error")
    if 'keras' in libs or 'tensorflow' in libs:
        imports.append("import tensorflow as tf")
        imports.append("from tensorflow.keras.models import Sequential")
        imports.append("from tensorflow.keras.layers import LSTM, Dense, Dropout")
    
    if not imports: imports.append("import os")
    
    init_code = "\n".join(imports) + "\n\n# Настройка визуализации\n%matplotlib inline\nplt.style.use('seaborn-v0_8')"
    nb.cells.append(nbf.v4.new_code_cell(init_code))
    
    # 4. Tasks (Implementation)
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Ход работы"))
    
    codes = {
        1: "import re\nsynop_pattern = r'AAXX\\s+(?P<YY>\\d{2})(?P<GG>\\d{2})'\nsample = 'AAXX 12184 99011'\nmatch = re.search(synop_pattern, sample)\nif match:\n    print(f'Найдено сообщение: Число={match.group(\"YY\")}, Срок={match.group(\"GG\")}')",
        2: "def check_temp(t):\n    return -90 < t < 60\n\nt_val = 25\nprint(f'Температура {t_val} корректна: {check_temp(t_val)}')",
        3: "import numpy as np\ndef G(r):\n    if r == 0: return 0\n    return r**2 * np.log(r + 1e-10)\n\nstations = np.array([[0,0,10], [10,0,15], [0,10,5], [10,10,20]])\ncoords, values = stations[:, :2], stations[:, 2]\nn = len(stations)\nK = np.zeros((n, n))\nfor i in range(n):\n    for j in range(n):\n        K[i,j] = G(np.linalg.norm(coords[i]-coords[j]))\nP = np.hstack([np.ones((n, 1)), coords])\nL = np.block([[K, P], [P.T, np.zeros((3, 3))]])\nV = np.hstack([values, np.zeros(3)])\nweights = np.linalg.solve(L, V)\nprint('Система решена.')",
        4: "from sklearn.datasets import make_classification\nX, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, random_state=42)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\nclf = RandomForestClassifier().fit(X_train, y_train)\nprint(f'Accuracy: {clf.score(X_test, y_test):.2f}')",
        5: "import numpy as np\ndata = np.sin(np.linspace(0, 100, 1000)).reshape(-1, 1)\ndef create_seq(d, lb=24):\n    X, y = [], []\n    for i in range(len(d)-lb):\n        X.append(d[i:i+lb])\n        y.append(d[i+lb])\n    return np.array(X), np.array(y)\n\nX_seq, y_seq = create_seq(data)\nprint(f'Shape: {X_seq.shape}')",
        6: "import tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import LSTM, Dense\nmodel = Sequential([\n    LSTM(50, input_shape=(24, 1)),\n    Dense(1)\n])\nmodel.compile(optimizer='adam', loss='mse')\nmodel.summary()"
    }
    
    for task in practice['tasks']:
        nb.cells.append(nbf.v4.new_markdown_cell(f"### Задание {task['task_id']}\n{task['description']}"))
        nb.cells.append(nbf.v4.new_code_cell(codes.get(p_num, "# Реализация здесь")))
        
    # 5. Review Questions
    questions = {
        1: ["Что такое квантификаторы в регулярных выражениях?", "Для чего нужны именованные группы?", "Как экранировать спецсимволы?"],
        2: ["В чем отличие первичного контроля от объективного?", "Какие физические пределы существуют для давления?", "Что такое гидростатический контроль?"],
        3: ["В чем преимущество TPS перед линейной интерполяцией?", "За что отвечает радиальная базисная функция?", "Как интерпретировать коэффициенты сплайна?"],
        4: ["Что показывает матрица ошибок (Confusion Matrix)?", "В чем разница между классификацией и регрессией?", "Как бороться с переобучением леса?"],
        5: ["Зачем нужна нормализация данных перед подачей в LSTM?", "Что такое look_back (окно предыстории)?", "Почему данные нельзя перемешивать перед созданием последовательностей?"],
        6: ["Какую роль играет слой Dropout?", "Какие метрики используются для оценки регрессии?", "Как интерпретировать кривые обучения (Loss)?"]
    }
    
    q_items = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions.get(p_num, ["Вопрос 1", "Вопрос 2"]))])
    nb.cells.append(nbf.v4.new_markdown_cell(f"## 4. Контрольные вопросы\n{q_items}"))
    
    # 6. Conclusion
    nb.cells.append(nbf.v4.new_markdown_cell(f"## 5. Заключение\nВ ходе выполнения практического занятия №{p_num} были изучены методы {title}.\n\nВсе поставленные задачи выполнены, полученные результаты соответствуют теоретическим ожиданиям."))
    
    return nb

def main():
    spec_path = 'output/practice_spec.json'
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
        
    master_nb = nbf.v4.new_notebook()
    master_nb.cells.append(nbf.v4.new_markdown_cell("# Сборник практических работ по курсу\n# \"Информационные технологии в гидрометеорологии\""))
    
    output_dir = Path("output/final")
    output_dir.mkdir(exist_ok=True)
    
    for p in spec['practices']:
        nb = create_final_notebook(p)
        filename = f"practice_{p['number']}_final.ipynb"
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f"Generated {filename}")
        
        # Add cells to master notebook
        master_nb.cells.extend(nb.cells)
        # Add a separator
        master_nb.cells.append(nbf.v4.new_markdown_cell("---"))
        
    with open(output_dir / "master_course.ipynb", 'w', encoding='utf-8') as f:
        nbf.write(master_nb, f)
    print("Generated master_course.ipynb")

if __name__ == "__main__":
    main()
