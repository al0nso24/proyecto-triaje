import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Carga de datos
df = pd.read_csv('dataset_triaje_sintetico.csv')
X = df.drop('prioridad', axis=1)
y = df['prioridad']

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

#Modelo principal (SVM entrenado con datos escalados)
modelo_svm = SVC(probability=True, random_state=42)
modelo_svm.fit(X_train_scaled, y_train)

#Función para imprimir métricas de Validación y Prueba
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

#X_val_scaled y X_test_scaled para el SVM
y_pred_test = evaluar_modelo(
    "Modelo Principal (SVM)", modelo_svm, X_val_scaled, y_val, X_test_scaled, y_test
)

print("\nReporte detallado por clase (SVM - Test):")
print(
    classification_report(
        y_test,
        y_pred_test,
        target_names=['0-Leve', '1-Moderado', '2-Crítico']
    )
)

#Generar y guardar Gráfico de Barras de la Matriz de Confusión (Evidencia)
cm = confusion_matrix(y_test, y_pred_test)

#Convirtiendo matriz de confusión a un DataFrame para graficar barras limpias
df_cm = pd.DataFrame(
    cm,
    index=['Real: Leve (0)', 'Real: Moderado (1)', 'Real: Crítico (2)'],
    columns=[
        'Pred: Leve (0)',
        'Pred: Moderado (1)',
        'Pred: Crítico (2)',
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
    'Evaluación por Clases - Matriz de Confusión (Modelo SVM)',
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
joblib.dump(modelo_svm, 'modelo_triaje.pkl')
print("Modelo serializado guardado como 'modelo_triaje.pkl'")

#Prueba de inferencia rápida en consola
paciente_nuevo = pd.DataFrame([{
    'edad': 68,
    'temperatura': 38.2,
    'frec_cardiaca': 125,
    'frec_respiratoria': 24,
    'saturacion_o2': 88.5,
    'presion_sistolica': 170,
    'dificultad_respiratoria': 1,
    'dolor_toracico': 1,
    'sangrado': 0,
    'perdida_conciencia': 0,
    'fiebre_persistente': 1,
    'vomitos': 0,
    'dolor_abdominal': 0,
    'dolor de cabeza': 0,
    'tos': 1,
    'malestar_general': 1,
}])

#Escalando también los datos del paciente nuevo antes de predecir con SVM
paciente_nuevo_scaled = scaler.transform(paciente_nuevo)
prediccion = modelo_svm.predict(paciente_nuevo_scaled)[0]

niveles = {
    0: 'Prioridad 0 - Leve',
    1: 'Prioridad 1 - Moderado',
    2: 'Prioridad 2 - Crítico (Atención Inmediata)',
}

print(f'\nPrueba con paciente nuevo -> Resultado: {niveles[prediccion]}')