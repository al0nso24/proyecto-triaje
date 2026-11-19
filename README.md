# TriageML – Sistema de triaje asistido por Machine Learning

Apoyo a la decisión de triaje: un modelo SVM sugiere la prioridad (Alta, Media o Baja) a partir de signos vitales y síntomas. El profesional confirma o modifica la sugerencia, con motivo obligatorio si la cambia, y cada decisión queda registrada con responsable, fecha y hora. El sistema es de apoyo y no reemplaza el criterio clínico.

Los datos son sintéticos. Los resultados no constituyen validación clínica.

## Contenido del repositorio

```text
proyecto-triaje/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── dataset_triaje_sintetico.csv
├── entrenar_triaje.py
├── modelo_triaje.pkl
├── model_card.json
├── matriz_confusion.png
├── backend/
│   ├── app.py
│   └── seed_demo.py
├── frontend/
└── docs/
    └── arquitectura.md
```

## Modelo de Machine Learning

- Dataset: `dataset_triaje_sintetico.csv`, 1,500 registros.
- Variables de entrada (16): edad, temperatura, frec_cardiaca, frec_respiratoria, saturacion_o2, presion_sistolica y 10 síntomas binarios (dificultad_respiratoria, dolor_toracico, sangrado, perdida_conciencia, fiebre_persistente, vomitos, dolor_abdominal, dolor de cabeza, tos, malestar_general).
- Variable objetivo `prioridad`: 0 = Leve (Baja), 1 = Moderado (Media), 2 = Crítico (Alta).
- Partición estratificada 70/15/15 (semilla 42). `StandardScaler` ajustado solo con los datos de entrenamiento.
- Baseline: Regresión Logística (`max_iter=1000`, `random_state=42`).
- Modelo principal: SVM (`SVC`, kernel RBF, `C=1.0`, `gamma='scale'`, `probability=True`, `random_state=42`).
- Se serializa un Pipeline (escalador + SVM) en `modelo_triaje.pkl` y la ficha del modelo en `model_card.json`.

| Modelo | Accuracy validación | Accuracy prueba | F1 ponderado validación | F1 ponderado prueba |
|---|---|---|---|---|
| Regresión Logística | 0.7467 | 0.6311 | 0.7471 | 0.6298 |
| SVM | 0.7333 | 0.6889 | 0.7338 | 0.6888 |

Recall del SVM en prueba (Leve / Moderado / Crítico): 0.737 / 0.690 / 0.612. Matriz de confusión del SVM en prueba (filas reales, columnas predichas):

```text
[[56, 19,  1],
 [24, 69,  7],
 [ 3, 16, 30]]
```

## Requisitos

- Python 3.12 o superior (probado con 3.12 en Linux y 3.14 en Windows)
- Node.js 22 LTS
- MySQL 8 (opcional; sin configurar se usa SQLite)

## Ejecución

Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python entrenar_triaje.py
```

En Linux o macOS se activa con `source .venv/bin/activate`.

Copia `.env.example` como `.env` y edita `DATABASE_URL` y `SECRET_KEY`. Para usar MySQL, crea antes la base:

```sql
CREATE DATABASE triageml CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Las tablas `usuario`, `atencion` y `evento` se crean solas al iniciar el backend. Si no existe `.env`, se usa SQLite.

```bash
python backend/seed_demo.py
python backend/app.py
```

La API queda en `http://127.0.0.1:5000`. En otra terminal:

```bash
cd frontend
npm install
npm run dev
```

La aplicación queda en `http://localhost:5173`.

## Cuentas de demostración

`python backend/seed_demo.py` crea 3 cuentas y 10 pacientes de ejemplo. Son solo para demostración local.

| Rol | Correo | Clave |
|---|---|---|
| Enfermería | enfermera@demo.pe | demo1234 |
| Médico | medico@demo.pe | demo1234 |
| Administrador | admin@demo.pe | demo1234 |

Si la base ya tiene pacientes, el script no hace nada; con `--agregar` añade los de demostración de todos modos.

## Funcionalidades

- Registro de paciente en 3 pasos: datos generales, signos vitales y síntomas, confirmación.
- Prioridad sugerida con probabilidades por clase y factores de referencia.
- Alerta de valores críticos: si hay signos vitales o síntomas en rango crítico y el modelo no sugiere Alta, se muestra una advertencia. No modifica la prioridad.
- Revisión profesional: confirmar o modificar la prioridad, con motivo obligatorio al modificar.
- Trazabilidad: historial de eventos con responsable, fecha y hora.
- Cola ordenada por prioridad vigente y hora de registro, con búsqueda y filtro.
- Roles: Enfermería y Médico registran y revisan; Administrador consulta.
- Tiempo de registro medido en la interfaz y mediana diaria en el panel de inicio.

## Endpoints

Requieren `Authorization: Bearer <token>`, salvo `health` y `auth`.

| Método | Ruta | Función |
|---|---|---|
| GET | /api/health | Estado de la API y del modelo |
| POST | /api/auth/registro | Crear cuenta |
| POST | /api/auth/login | Iniciar sesión |
| GET | /api/auth/yo | Usuario de la sesión |
| POST | /api/atenciones | Registrar paciente y obtener la prioridad sugerida |
| GET | /api/atenciones/{id} | Detalle |
| PATCH | /api/atenciones/{id}/revision | Confirmar o modificar la prioridad |
| PATCH | /api/atenciones/{id}/estado | En espera, En atención o Finalizado |
| GET | /api/atenciones/{id}/historial | Trazabilidad |
| GET | /api/cola | Cola de atención (`q`, `prioridad`) |
| GET | /api/resumen | Contadores del día |

## Limitaciones

- Datos sintéticos: las métricas solo comparan modelos y no demuestran seguridad clínica.
- El SVM detecta 30 de 49 casos críticos en la prueba (recall 0.612); la revisión profesional es obligatoria.
- Los factores mostrados son reglas de referencia y no la explicación interna del SVM.
- Cualquier persona puede elegir su rol al crear una cuenta.
- Despliegue solo local.
