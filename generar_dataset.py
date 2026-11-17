import numpy as np
import pandas as pd

np.random.seed(42)
n = 2500

#Datos del paciente
edad = np.random.randint(18, 85, size=n)
frecuencia_cardiaca = np.random.normal(85, 18, size=n).clip(45,160).astype(int)
presion_sistolica = np.random.normal(122, 22, size=n).clip(70, 210).astype(int)
saturacion_oxigeno = np.random.normal(96, 3.5, size=n).clip(78, 100).round(1)
temperatura = np.random.normal(37.1, 0.9, size=n).clip(35.0, 41.2).round(1)
dolor_pecho = np.random.choice([0, 1], size=n, p=[0.82, 0.18])
dificultad_respiratoria = np.random.choice([0, 1], size=n, p=[0.80, 0.20])

#Cálculo de riesgo clínico para asignar la prioridad de triaje
score = (
    (saturacion_oxigeno < 90) * 4
    + (dolor_pecho == 1) * 3
    + (dificultad_respiratoria == 1) * 3
    + (frecuencia_cardiaca > 120) * 2
    + (presion_sistolica > 165) * 2
    + (presion_sistolica < 90) * 3
    + (temperatura > 39.0) * 1.5
    + (edad > 65) * 1
    #Ruido clínico realista
    + np.random.normal(0, 1.1, size=n)
)

#Crítico (rojo)
#Moderado (amarillo)
#Leve (verde)
prioridad_triaje = np.where(
    score >= 5.5, 1, np.where(score >= 2.5, 2, 3)
)

#Cuadro
df = pd.DataFrame({
    "Edad": edad,
    "Frecuencia_cardiaca": frecuencia_cardiaca,
    "Presion_sistolica": presion_sistolica,
    "Saturacion_oxigeno": saturacion_oxigeno,
    "Temperatura": temperatura,
    "Dolor_pecho": dolor_pecho,
    "Dificultad_respiratoria": dificultad_respiratoria,
    "Prioridad_triaje": prioridad_triaje
})

df.to_csv('dataset_triaje.csv', index=False)
print('Dataset generado con éxito:', df.shape)
print(df['prioridad_triaje'].value_counts())