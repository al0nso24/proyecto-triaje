import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Carga de datos
df = pd.read_csv('dataset_triaje.csv')
X = df.drop('Prioridad_triaje', axis=1)
y = df['Prioridad_triaje']

#Partición Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

#Escalamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

#Modelo Baseline: Regresión Logística
baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train_scaled, y_train)

#Modelo principal
modelo_rf = RandomForestClassifier(
    n_estimators=150, max_depth=10, random_state=42
)
modelo_rf.fit(X_train, y_train)

#Función para imprimir métricad de Validación y Prueba
def evaluar_modelo(nombre, modelo, X_v, y_v, X_t, y_t):
    pred_v = modelo.predict(X_v)
    pred_t = modelo.predict(X_t)
    print(f'\n=== {nombre} ===')
    print(
       f'Accuracy  -> Val: {accuracy_score(y_v, pred_v):.4f} | Test:'
       f' {accuracy_score(y_t, pred_t):.4f}'
    )
    print(
        'Precision -> Val:'
        f" {precision_score(y_v, pred_v, average='weighted'):.4f} | Test:"
        f" {precision_score(y_t, pred_t, average='weighted'):.4f}"
    )
    print(
        'Recall    -> Val:'
        f" {recall_score(y_v, pred_v, average='weighted'):.4f} | Test:"
        f" {recall_score(y_t, pred_t, average='weighted'):.4f}"
    )
    print(
        'F1-Score  -> Val:'
        f" {f1_score(y_v, pred_v, average='weighted'):.4f} | Test:"
        f" {f1_score(y_t, pred_t, average='weighted'):.4f}"
    )
    return pred_t

evaluar_modelo(
    "Baseline (Regresión Logística)",
    baseline,
    X_val_scaled,
    y_val,
    X_test_scaled,
    y_test
)

y_pred_test = evaluar_modelo(
    "Modelo Principal (Random Forest)", modelo_rf, X_val, y_val, X_test, y_test
)

print("\nReporte detallado por clase (Random Forest - Test):")
print(
    classification_report(
        y_test, y_pred_test, target_names=['1-Crítico', '2-Moderado', '3-Leve']
    )
)

#Generar y guardar Gráfico de Barras de la Matriz de Confusión (Evidencia)
cm = confusion_matrix(y_test, y_pred_test)

#Convirtiendo matriz de confusión a un DataFrame para graficar barras limpias
df_cm = pd.DataFrame(
    cm,
    index=['Real: Crítico (1)', 'Real: Moderado (2)', 'Real: Leve (3)'],
    columns=[
        'Pred: Crítico (1)',
        'Pred: Moderado (2)',
        'Pred: Leve (3)',
    ],
).reset_index()

df_cm_melted = df_cm.melt(
    id_vars='index', var_name='Prediccion', value_name='Cantidad'
)

df_cm_melted.rename(columns={'index': 'Valor_Real'}, inplace=True)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=df_cm_melted,
    x='Valor_Real',
    y='Cantidad',
    hue='Prediccion',
    palette='Blues',
)

plt.title(
    'Evaluación por Clases - Matriz de Confusión (Modelo Random Forest)',
    fontsize=11,
    weight='bold',
)

plt.xlabel('Categoría Real del Paciente', fontsize=10)
plt.ylabel('Cantidad de Pacientes', fontsize=10)
plt.legend(title='Resultado del Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300)
print("Gráfico de barras guardado como 'matriz_confusion.png'")

#Serialización del modelo
joblib.dump(modelo_rf, 'modelo_triaje.pkl')
print("Modelo serializado guardado como 'modelo_triaje.pkl'")

#Prueba de inferencia rápida en consola
paciente_nuevo = pd.DataFrame([{
    'Edad': 68,
    'Frecuencia_cardiaca': 125,
    'Presion_sistolica': 170,
    'Saturacion_oxigeno': 88.5,
    'Temperatura': 38.2,
    'Dolor_pecho': 1,
    'Dificultad_respiratoria': 1,
}])

prediccion = modelo_rf.predict(paciente_nuevo)[0]
niveles = {
    1: 'Prioridad 1 - Crítico (Atención Inmediata)',
    2: 'Prioridad 2 - Moderado',
    3: 'Prioridad 3 - Leve',
}

print(f'\nPrueba con paciente nuevo -> Resultado: {niveles[prediccion]}')