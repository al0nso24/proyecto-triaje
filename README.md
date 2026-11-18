# 🏥 Sistema Predictivo de Triaje Médico (ML)

Modelo de aprendizaje automático desarrollado para clasificar y priorizar pacientes en áreas de urgencias según la gravedad de sus signos vitales y síntomas clínicos.

---

## 📋 Descripción del Proyecto
Este repositorio contiene el pipeline completo de generación de datos sintéticos, preprocesamiento, entrenamiento, validación y serialización de un modelo de clasificación multiclase para triaje médico. 

El sistema clasifica a cada paciente en tres niveles de atención:
- 🔴 **Prioridad 1 — Crítico:** Atención médica inmediata.
- 🟡 **Prioridad 2 — Moderado:** Atención urgente con observación.
- 🟢 **Prioridad 3 — Leve:** Consulta médica estándar (no urgente).

---

## 📂 Estructura del Proyecto

```text
proyecto-triaje/
├── generar_dataset.py     # Script de generación de datos sintéticos con lógica clínica
├── dataset_triaje.csv     # Dataset generado (2,500 registros clínicos)
├── entrenar_triaje.py     # Pipeline de entrenamiento, validación, métricas y exportación
├── modelo_triaje.pkl      # Modelo SVM serializado listo para inferencia
├── matriz_confusion.png   # Gráfico de evaluación por clases (generado automáticamente)
└── README.md              # Documentación del proyecto
```

---

## 🛠️ Tecnologías y Librerías
- **Lenguaje:** Python 3.x
- **Manipulación de datos:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn` (`SVC`, `LogisticRegression`, `StandardScaler`)
- **Visualización:** `matplotlib`, `seaborn`
- **Serialización:** `joblib`

---

## 🚀 Instalación y Ejecución

1. **Clonar o descargar el proyecto** y abrir la carpeta en tu editor (ej. VS Code).

2. **Instalar las dependencias necesarias:**
   ```bash
   pip install pandas numpy scikit-learn joblib matplotlib seaborn
   ```

3. **Generar el conjunto de datos sintético:**
   ```bash
   python generar_dataset.py
   ```
   *(Genera el archivo `dataset_triaje.csv` con 2,500 registros).*

4. **Entrenar, evaluar y exportar el modelo:**
   ```bash
   python entrenar_triaje.py
   ```
   *(Entrena el baseline y el modelo principal con escalamiento, imprime las métricas en consola, guarda `matriz_confusion.png` y exporta `modelo_triaje.pkl`).*

---

## 📊 Resultados Experimentales (Conjunto de Prueba)

Se utilizó una partición estratificada de **70% Train / 15% Validation / 15% Test** con semilla fija (`random_state=42`) y estandarización de variables (`StandardScaler`):

| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Regresión Logística)** | 79.47% | 0.7909 | 0.7947 | 0.7886 |
| **Modelo Principal (SVM)** | **83.20%** | **0.8369** | **0.8320** | **0.8315** |