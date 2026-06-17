# ✈️ Proyección de Tráfico Aéreo con LSTM

Sistema de pronóstico de tráfico aéreo basado en Deep Learning. El modelo utiliza una red neuronal recurrente LSTM (Long Short-Term Memory) para predecir el flujo mensual de pasajeros a partir de datos históricos, aplicando ingeniería de variables temporales para capturar patrones estacionales.

---

## 📂 Dataset

**Airlines Traffic Passenger Statistics** — disponible en [Kaggle](https://www.kaggle.com/datasets/thedevastator/airlines-traffic-passenger-statistics)

Contiene el conteo mensual de pasajeros ajustado, segmentado por aerolínea, región geográfica y período de actividad.

---

## 🛠️ Metodología

### 1. Análisis Exploratorio (EDA)
- Limpieza de nulos en columnas de código IATA de aerolíneas operadoras y publicadas
- Análisis de la distribución de regiones geográficas (GEO Region) y años disponibles
- Visualización de la serie histórica de pasajeros ajustados por mes

### 2. Feature Engineering
Se construyeron variables temporales para capturar estacionalidad y tendencia:

| Variable | Descripción |
|---|---|
| Trimestre | Trimestre del año extraído de la fecha |
| Mes | Mes numérico extraído de la fecha |
| Month_Sin, Month_Cos | Codificación cíclica del mes (seno/coseno) para preservar la continuidad estacional |
| Nav | Variable dummy para temporada navideña (diciembre-enero) |
| Verano | Variable dummy para temporada de verano (junio-agosto) |
| Lag_1M, Lag_2M, Lag_3M, Lag_4M, Lag_6M, Lag_12M | Valores rezagados del tráfico total para capturar dependencia temporal |

### 3. Preprocesamiento
- Normalización de variables con MinMaxScaler
- División temporal 80/20 (entrenamiento/prueba), respetando el orden cronológico
- Construcción de secuencias 3D (ventana de 12 meses de lookback) para alimentar la LSTM

### 4. Arquitectura del Modelo

Input (12 timesteps, N features) → LSTM (50 unidades, return_sequences=True) → Dropout (0.3) → LSTM (50 unidades) → Dropout (0.3) → Dense (1 unidad) → Predicción

- Optimizador: Adam (learning_rate=0.001)
- Función de pérdida: MSE
- Entrenamiento: 100 épocas, batch size 32, 10% de validación

### 5. Evaluación
El modelo se evalúa sobre el conjunto de prueba con las predicciones desnormalizadas a su escala original (número de pasajeros):
- RMSE — Error en pasajeros
- MAE — Error absoluto promedio
- MAPE — Error porcentual promedio

### 6. Visualización
- Comparación gráfica entre tráfico real y predicción LSTM en el conjunto de prueba
- Visualización del histórico completo junto con las predicciones para evaluar el ajuste del modelo en contexto

---

## 🚀 Cómo ejecutar

### 1. Clona el repositorio

git clone https://github.com/RichLD/Proyecci-n-de-Tr-fico-A-reo-con-LSTM.git
cd Proyecci-n-de-Tr-fico-A-reo-con-LSTM

### 2. Instala las dependencias

pip install pandas numpy matplotlib seaborn cufflinks scikit-learn tensorflow kagglehub

### 3. Ejecuta el script

python "Proyección de Tráfico Aereo con LSTM.py"

> El dataset se descarga automáticamente desde Kaggle con kagglehub. Necesitas tener configuradas tus credenciales de Kaggle en ~/.kaggle/kaggle.json.

---

## 📦 Dependencias principales

pandas, numpy, matplotlib, seaborn, cufflinks, scikit-learn, tensorflow, kagglehub

---

## 🗂️ Estructura del proyecto

Proyección de Tráfico Aereo con LSTM.py
README.md

---

## ⚠️ Notas y limitaciones

- El dataset cubre un rango histórico limitado, lo que puede afectar la capacidad del modelo para generalizar ante eventos atípicos (crisis sanitarias, cambios regulatorios, etc.)
- Las variables dummy de temporada (Nav, Verano) están definidas manualmente y podrían refinarse incorporando más eventos relevantes (Spring Break, Acción de Gracias).

---

## 👤 Autor

**Ricardo López Degollado**
[LinkedIn](https://linkedin.com/in/tu-perfil) · [GitHub](https://github.com/RichLD)
