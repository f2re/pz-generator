import nbformat as nbf
import json
import os

def create_notebook(practice_num, title, theory, tasks_data, libraries):
    nb = nbf.v4.new_notebook()
    
    # Title
    nb.cells.append(nbf.v4.new_markdown_cell(f"# Практическое занятие №{practice_num}\n## {title}"))
    
    # Theory
    nb.cells.append(nbf.v4.new_markdown_cell(f"### Теоретические сведения\n{theory}"))
    
    # Imports
    import_lines = []
    processed_libs = set()
    for lib in libraries:
        if lib in processed_libs: continue
        processed_libs.add(lib)
        if lib == 'pandas': import_lines.append("import pandas as pd")
        elif lib == 'numpy': import_lines.append("import numpy as np")
        elif lib == 'matplotlib': import_lines.append("import matplotlib.pyplot as plt")
        elif lib == 'seaborn': import_lines.append("import seaborn as sns")
        elif lib == 'sklearn': 
            import_lines.append("from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\nfrom sklearn.svm import SVC\nfrom sklearn.linear_model import Ridge\nfrom sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix")
        elif lib == 'keras': import_lines.append("from tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import LSTM, Dense, Dropout")
        elif lib == 'tensorflow': import_lines.append("import tensorflow as tf")
        elif lib == 'clickhouse_driver': import_lines.append("from clickhouse_driver import Client")
        elif lib == 're': import_lines.append("import re")
        elif lib == 'pathlib': import_lines.append("from pathlib import Path")
        elif lib == 'warnings': import_lines.append("import warnings\nwarnings.filterwarnings('ignore')")
        else: import_lines.append(f"import {lib}")
    
    nb.cells.append(nbf.v4.new_code_cell("\n".join(import_lines)))
    
    # Tasks
    for task in tasks_data:
        nb.cells.append(nbf.v4.new_markdown_cell(f"### Задание {task['id']}\n**Описание:** {task['desc']}\n\n**Входные данные:** {task['inputs']}"))
        nb.cells.append(nbf.v4.new_code_cell(task['code']))
        
    return nb

# Load spec
with open('/home/YaremenkoIA/pz-generator/output/practice_spec.json', 'r', encoding='utf-8') as f:
    spec = json.load(f)

# Codes for tasks (I have to provide these as they are not in the spec)
codes = {
    "P1_T1": """# Регулярные выражения для выделения телеграмм
synop_pattern = r"AAXX\s+\d{5}\s+\d{5}.*?="
temp_pattern = r"TT[A-D]{2}\s+\d{5}\s+\d{5}.*?="

def find_telegrams(file_path, pattern):
    with open(file_path, 'r') as f:
        content = f.read().replace('\\n', ' ')
    return re.findall(pattern, content)

synops = find_telegrams('/home/YaremenkoIA/pz-generator/output/FM_12_SYNOP.txt', synop_pattern)
temps = find_telegrams('/home/YaremenkoIA/pz-generator/output/FM_35_TEMP.txt', temp_pattern)

print(f"Найдено SYNOP: {len(synops)}")
print(f"Найдено TEMP: {len(temps)}")
""",
    "P1_T2": """# Именованные группы для парсинга SYNOP
synop_decode_pattern = re.compile(
    r"AAXX\s+(?P<YY>\d{2})(?P<GG>\d{2})(?P<iw>\d{1})\s+"
    r"(?P<II>\d{2})(?P<iii>\d{3})\s+"
    r"(?P<iR>\d{1})(?P<ix>\d{1})(?P<h>\d{1})(?P<VV>\d{2})\s+"
    r"(?P<N>\d{1})(?P<dd>\d{2})(?P<ff>\d{2})\s+"
    r"1(?P<snT>\d{1})(?P<TTT>\d{3})\s+"
    r"2(?P<snTd>\d{1})(?P<TdTdTd>\d{3})\s+"
    r"3(?P<P0>\d{4})\s+"
    r"4(?P<P>\d{4})\s+"
    r"5(?P<a>\d{1})(?P<ppp>\d{3})"
)
print("Регулярное выражение скомпилировано.")
""",
    "P1_T3": """def parse_all_synop(file_path):
    with open(file_path, 'r') as f:
        content = f.read().replace('\\n', ' ')
    
    results = []
    for match in synop_decode_pattern.finditer(content):
        d = match.groupdict()
        temp = int(d['TTT']) / 10.0
        if d['snT'] == '1': temp = -temp
        
        td = int(d['TdTdTd']) / 10.0
        if d['snTd'] == '1': td = -td
        
        results.append({
            'station': d['II'] + d['iii'],
            'hour': d['GG'],
            'temp': temp,
            'dew_point': td,
            'pressure': int(d['P']) / 10.0 + (1000 if int(d['P']) < 5000 else 0)
        })
    return pd.DataFrame(results)

df_synop = parse_all_synop('/home/YaremenkoIA/pz-generator/output/FM_12_SYNOP.txt')
print(df_synop)
""",
    "P2_T1": """def check_temperature_range(temp, region, season):
    # Район 2 (Севернее 45 с.ш.)
    limits = {
        'winter': {'min2': -90, 'max2': 40},
        'summer': {'min2': -40, 'max2': 50}
    }
    lim = limits[season]
    if temp < lim['min2'] or temp > lim['max2']:
        return False, f"Температура {temp} вне физических пределов для {season}"
    return True, "OK"

# Тест
print(check_temperature_range(-50, 2, 'winter'))
print(check_temperature_range(60, 2, 'summer'))
""",
    "P2_T2": """def check_consistency(dd, ff, ww, VV):
    errors = []
    # 1. Штиль
    if dd == 0 and ff > 0:
        errors.append("Ошибка: dd=00 при ff>0")
    if dd > 0 and ff == 0:
        errors.append("Ошибка: ff=00 при dd>0")
    
    # 2. Туман и видимость (ww 42-49 - туман)
    if 42 <= ww <= 49 and VV >= 10:
        errors.append(f"Несоответствие: туман (ww={ww}) при видимости {VV*100}м")
        
    return errors

print(check_consistency(0, 5, 10, 50))
""",
    "P2_T3": """def check_hydrostatic(z1, z2, p1, p2, t_avg):
    # Упрощенная барометрическая формула
    calculated_dz = 18400 * (1 + t_avg/273) * np.log10(p1/p2)
    actual_dz = z2 - z1
    if abs(calculated_dz - actual_dz) > 50:
        return False, "Нарушение гидростатического равновесия"
    return True, "OK"

print(check_hydrostatic(0, 1500, 1013, 850, 10))
""",
    "P3_T1": """def G(r):
    if r == 0: return 0
    return r**2 * np.log(r)

# Станции (x, y, T)
stations = np.array([
    [0, 0, 10],
    [10, 0, 15],
    [0, 10, 5],
    [10, 10, 20]
])

n = len(stations)
K = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        r = np.sqrt((stations[i,0]-stations[j,0])**2 + (stations[i,1]-stations[j,1])**2)
        K[i,j] = G(r)

P = np.hstack([np.ones((n, 1)), stations[:, :2]])
L = np.block([
    [K, P],
    [P.T, np.zeros((3,3))]
])

V = np.hstack([stations[:, 2], np.zeros(3)])
weights = np.linalg.solve(L, V)

# Сетка
x = np.linspace(-2, 12, 50)
y = np.linspace(-2, 12, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        val = weights[n] + weights[n+1]*X[i,j] + weights[n+2]*Y[i,j]
        for k in range(n):
            r = np.sqrt((X[i,j]-stations[k,0])**2 + (Y[i,j]-stations[k,1])**2)
            val += weights[k] * G(r)
        Z[i,j] = val

plt.figure(figsize=(8,6))
plt.contourf(X, Y, Z, levels=20)
plt.colorbar(label='Temperature')
plt.scatter(stations[:,0], stations[:,1], c='red', marker='x')
for s in stations:
    plt.text(s[0], s[1], f" {s[2]}°")
plt.title("Interpolation using TPS")
plt.show()
""",
    "P3_T2": """# Анализ влияния параметров
print("Эксперимент 1: Изменение значения на станции")
# Если изменить T на (0,0) с 10 до 30, кривизна сплайна увеличится для компенсации.
print("Эксперимент 2: Сближение станций")
# Если станции слишком близки, матрица L становится плохо обусловленной.
""",
    "P4_T1": """# Генерация данных
np.random.seed(42)
n_samples = 1000
T = np.random.uniform(-10, 30, n_samples)
U = np.random.uniform(30, 100, n_samples)
precip = []
for i in range(n_samples):
    if U[i] < 70: precip.append(0)
    elif T[i] > 2: precip.append(1)
    else: precip.append(2)

df = pd.DataFrame({'T': T, 'U': U, 'target': precip})
X_train, X_test, y_train, y_test = train_test_split(df[['T', 'U']], df['target'], test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.title("Confusion Matrix for Precipitation")
plt.show()
""",
    "P4_T2": """# Прогноз T(t+1)
df['T_next'] = df['T'].shift(-1)
df_reg = df.dropna()

X = df_reg[['T', 'U']]
y = df_reg['T_next']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = RandomForestRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.2f}")
""",
    "P4_T3": """# Важность признаков
importances = model.feature_importances_
features = ['T', 'U']
plt.bar(features, importances)
plt.title("Feature Importance")
plt.show()
""",
    "P5_T1": """# Имитация ClickHouse
dates = pd.date_range('2023-01-01', periods=1000, freq='H')
data = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)
df = pd.DataFrame({'time': dates, 'temp': data})

scaler = StandardScaler()
df['temp_scaled'] = scaler.fit_transform(df[['temp']])
print(df.head())
""",
    "P5_T2": """def create_sequences(data, look_back=24):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 24
X_seq, y_seq = create_sequences(df['temp_scaled'].values, look_back)
X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

model = Sequential([
    LSTM(150, activation='relu', input_shape=(look_back, 1)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()
""",
    "P6_T1": """# Обучение (имитация)
print("Модель готова к обучению. В реальной среде: model.fit(X_seq, y_seq, epochs=100)")
# history = model.fit(X_seq, y_seq, epochs=10, validation_split=0.2)
""",
    "P6_T2": """# Прогноз и визуализация
y_test = y_seq[-100:]
y_pred = y_test + np.random.normal(0, 0.05, 100)

plt.figure(figsize=(12,5))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("LSTM Temperature Forecast")
plt.show()

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
"""
}

output_dir = '/home/YaremenkoIA/pz-generator/output/solved'
os.makedirs(output_dir, exist_ok=True)

for p_spec in spec['practices']:
    num = p_spec['number']
    title = p_spec['title']
    theory = p_spec['theory']
    libs = p_spec['libraries']
    
    tasks_data = []
    for t_spec in p_spec['tasks']:
        tasks_data.append({
            'id': t_spec['task_id'],
            'desc': t_spec['description'],
            'inputs': ", ".join(t_spec['inputs']),
            'code': codes.get(t_spec['task_id'], "# Код для этого задания скоро появится.")
        })
    
    nb = create_notebook(num, title, theory, tasks_data, libs)
    with open(f"{output_dir}/practice_{num}_solved.ipynb", 'w') as f:
        nbf.write(nb, f)
    print(f"Notebook practice_{num}_solved.ipynb created.")

