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
- Análisis de la distribución de regiones geográficas (`GEO Region`) y años disponibles
- Visualización de la serie histórica de pasajeros ajustados por mes

### 2. Feature Engineering
Se construyeron variables temporales para capturar estacionalidad y tendencia:

| Variable | Descripción |
|---|---|
| `Trimestre` | Trimestre del año extraído de la fecha |
| `Mes` | Mes numérico extraído de la fecha |
| `Month_Sin`, `Month_Cos` | Codificación cíclica del mes (seno/coseno) para preservar la continuidad estacional |
| `Nav` | Variable dummy para temporada navideña (diciembre-enero) |
| `Verano` | Variable dummy para temporada de verano (junio-agosto) |
| `Lag_1M`, `Lag_2M`, `Lag_3M`, `Lag_4M`, `Lag_6M`, `Lag_12M` | Valores rezagados del tráfico total para capturar dependencia temporal |

### 3. Preprocesamiento
- Normalización de variables con `MinMaxScaler`
- División temporal 80/20 (entrenamiento/prueba), respetando el orden cronológico
- Construcción de secuencias 3D (ventana de 12 meses de lookback) para alimentar la LSTM

### 4. Arquitectura del Modelo
