import streamlit as st
import pandas as pd
import numpy as np
import random
import datetime # Para simular datos a lo largo del tiempo
import time # Para simular el tiempo real
import seaborn as sns # Importado pero no usado, si no lo necesitas, puedes quitarlo

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Monitoreo Financiero Petroperú (Simulado)", page_icon="📈", layout="wide")

st.title("📈 Monitoreo Financiero Petroperú (Simulado con IA Conceptual)")

st.markdown(
    """
    Esta aplicación simula un **panel de monitoreo en tiempo real** para el área financiera
    de Petroperú. Muestra métricas clave, tendencias y detección de anomalías (simuladas),
    además de conceptualizar cómo la **Inteligencia Artificial** potenciaría un sistema real.

    **Disclaimer:** Todos los datos son generados aleatoriamente y no representan información real de Petroperú.
    """
)
st.write("---")

# --- FUNCIÓN PARA GENERAR DATOS SIMULADOS ---
# Usaremos st.cache_data para que los datos no cambien en cada interacción
# pero la "actualización en tiempo real" la manejaremos con una lógica de refresco
@st.cache_data(ttl=5) # Los datos se refrescarán cada 5 segundos para simular "tiempo real"
def generate_financial_data():
    current_time = pd.Timestamp.now()
    # Generar datos para las últimas 24 horas (intervalo de 1 hora)
    dates = pd.date_range(end=current_time, periods=24, freq='H')

    # Simulación de precios del petróleo (Brent)
    base_price = 80
    oil_prices = [base_price + random.uniform(-5, 5) for _ in range(len(dates))]
    oil_prices = [max(70, min(90, p)) for p in oil_prices] # Limitar entre 70 y 90

    # Simulación de Ingresos y Gastos (millones de USD)
    # Los ingresos y gastos pueden fluctuar y tener una relación con el precio del petróleo
    revenues = [p * random.uniform(1.2, 1.5) * random.uniform(50, 100) / 1000 for p in oil_prices]
    expenses = [p * random.uniform(0.8, 1.1) * random.uniform(50, 100) / 1000 for p in oil_prices]

    # Añadir una anomalía simulada en un punto aleatorio para las últimas horas
    if random.random() < 0.3: # 30% de probabilidad de anomalía
        anomaly_idx = random.randint(5, len(dates) - 2) # Un índice al azar, no en los extremos
        if random.random() < 0.5: # Anomalía positiva (ej. pico de ingresos inesperado)
            revenues[anomaly_idx] *= random.uniform(1.5, 2.5)
            st.session_state.anomaly_alert = f"ALERTA: Pico inusual de ingresos en {dates[anomaly_idx].strftime('%H:%M')}!"
        else: # Anomalía negativa (ej. caída de ingresos o pico de gastos)
            revenues[anomaly_idx] *= random.uniform(0.5, 0.7)
            expenses[anomaly_idx] *= random.uniform(1.5, 2.0)
            st.session_state.anomaly_alert = f"ALERTA: Desviación significativa de fondos/ingresos en {dates[anomaly_idx].strftime('%H:%M')}!"
    else:
        st.session_state.anomaly_alert = "Monitoreo normal. No se detectaron anomalías."

    df = pd.DataFrame({
        "Fecha": dates,
        "Precio Petróleo Brent (USD/barril)": oil_prices,
        "Ingresos (MM USD)": revenues,
        "Gastos (MM USD)": expenses
    })
    df["Beneficio (MM USD)"] = df["Ingresos (MM USD)"] - df["Gastos (MM USD)"]
    df = df.set_index("Fecha")
    return df

# Inicializar estado de la sesión si no existe
if 'anomaly_alert' not in st.session_state:
    st.session_state.anomaly_alert = "Iniciando monitoreo..."

# Obtener los datos (se refrescarán cada 5 segundos debido al ttl de st.cache_data)
financial_df = generate_financial_data()

# --- VISIÓN GENERAL: MÉTRICAS CLAVE ---
st.header("📊 Visión General del Desempeño Financiero")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Ingresos (última hora)",
        value=f"{financial_df['Ingresos (MM USD)'].iloc[-1]:.2f} MM USD"
    )
    st.write(f"Promedio 24h: {financial_df['Ingresos (MM USD)'].mean():.2f} MM USD")

with col2:
    st.metric(label="Gastos (última hora)",
              value=f"{financial_df['Gastos (MM USD)'].iloc[-1]:.2f} MM USD")
    st.write(f"Promedio 24h: {financial_df['Gastos (MM USD)'].mean():.2f} MM USD")

with col3:
    st.metric(label="Beneficio (última hora)",
              value=f"{financial_df['Beneficio (MM USD)'].iloc[-1]:.2f} MM USD",
              delta=f"{(financial_df['Beneficio (MM USD)'].iloc[-1] - financial_df['Beneficio (MM USD)'].iloc[-2]):.2f} MM USD vs. hora anterior")
    st.write(f"Promedio 24h: {financial_df['Beneficio (MM USD)'].mean():.2f} MM USD")

with col4:
    st.metric(label="Precio Brent (última hora)",
              value=f"{financial_df['Precio Petróleo Brent (USD/barril)'].iloc[-1]:.2f} USD")
    st.write(f"Promedio 24h: {financial_df['Precio Petróleo Brent (USD/barril)'].mean():.2f} USD")

st.write("---")

# --- TENDENCIAS CLAVE ---
st.header("📈 Tendencias Financieras (Últimas 24 Horas)")

# Gráfico de Ingresos y Gastos
fig_fin, ax_fin = plt.subplots(figsize=(12, 5))
ax_fin.plot(financial_df.index, financial_df["Ingresos (MM USD)"], label="Ingresos", color="green")
ax_fin.plot(financial_df.index, financial_df["Gastos (MM USD)"], label="Gastos", color="red")
ax_fin.set_title("Ingresos y Gastos (MM USD)")
ax_fin.set_xlabel("Hora")
ax_fin.set_ylabel("Monto (MM USD)")
ax_fin.tick_params(axis='x', rotation=45)
ax_fin.legend()
plt.tight_layout()
st.pyplot(fig_fin)
plt.close(fig_fin) # Importante para liberar memoria

# Gráfico de Precio del Petróleo
fig_oil, ax_oil = plt.subplots(figsize=(12, 5))
ax_oil.plot(
    financial_df.index, financial_df["Precio Petróleo Brent (USD/barril)"], label="Precio Brent", color="blue")
ax_oil.set_title("Precio Petróleo Brent (USD/barril)")
ax_oil.set_xlabel("Hora")
ax_oil.set_ylabel("Precio (USD)")
ax_oil.tick_params(axis='x', rotation=45)
ax_oil.legend()
plt.tight_layout()
st.pyplot(fig_oil)
plt.close(fig_oil) # Importante para liberar memoria

st.write("---")

# --- DETECCIÓN DE ANOMALÍAS (SIMULADA POR IA) ---
st.header("🚨 Detección de Anomalías (AI Simulada)")
st.warning(st.session_state.anomaly_alert) # Muestra la alerta guardada en session_state

st.markdown(
    """
    *Aquí, un sistema de IA real analizaría patrones históricos en tiempo real para identificar
    desviaciones significativas en los datos financieros que puedan indicar fraude, errores operativos,
    cambios bruscos en el mercado o eventos inesperados. Los modelos podrían incluir:*

    - **Algoritmos de Series Temporales:** ARIMA, Prophet, Holt-Winters para predecir valores esperados.
    - **Modelos de Aprendizaje No Supervisado:** K-Means, Isolation Forest, One-Class SVM para detectar outliers.
    - **Redes Neuronales Recurrentes (RNNs):** Para aprender patrones complejos en secuencias de datos.
    """
)

st.write("---")

# --- CONCEPTO DE IA EN EL MONITOREO FINANCIERO ---
st.header("🧠 Cómo una IA Real Mejoraría este Monitoreo")

st.markdown(
    """
    Un sistema de IA real iría mucho más allá de las visualizaciones básicas, integrando modelos
    de Machine Learning avanzados para:

    1.  **Predicción de Precios y Volatilidad:**
        *   Modelos de Series Temporales (ARIMA, Prophet, LSTM) para pronosticar precios de petróleo, tipo de cambio y demanda de productos con alta precisión.
        *   Análisis de factores macroeconómicos, geopolíticos y patrones históricos.
        *   **Beneficio:** Mejora la toma de decisiones de cobertura, compra/venta y planificación estratégica.

    2.  **Detección de Fraude y Anomalías:**
        *   Algoritmos de detección de valores atípicos (Isolation Forest, One-Class SVM) para identificar transacciones inusuales, patrones de gastos anómalos o desviaciones inesperadas en ingresos.
        *   Monitoreo de auditorías y conciliaciones automáticas.
        *   **Beneficio:** Minimiza riesgos financieros, previene pérdidas por fraude y errores operativos.

    3.  **Análisis de Riesgo Cuantitativo:**
        *   Modelos de riesgo de crédito, riesgo de mercado y riesgo operativo basados en datos históricos y factores externos.
        *   Simulaciones Monte Carlo para evaluar el impacto de diferentes escenarios.
        *   **Beneficio:** Permite una gestión de riesgo proactiva y una asignación de capital más eficiente.

    4.  **Optimización de Operaciones y Logística:**
        *   Modelos de optimización para la cadena de suministro, refinación y distribución, considerando costos, demanda y capacidad.
        *   **Beneficio:** Reduce costos operativos, mejora la eficiencia y maximiza la rentabilidad.

    5.  **Análisis de Sentimientos de Noticias Financieras:**
        *   Procesamiento de Lenguaje Natural (NLP) para analizar noticias, informes y redes sociales, identificando eventos que podrían afectar los mercados o la reputación de la empresa.
        *   **Beneficio:** Alertas tempranas sobre eventos críticos y comprensión del entorno de mercado.

    La IA transformaría este panel en una herramienta inteligente capaz de ofrecer insights predictivos, alertar proactivamente sobre riesgos y optimizar las operaciones financieras de Petroperú.
    """
)

st.write("---")
st.markdown("Desarrollado con 🚀 en Python y Streamlit para fines demostrativos y simulados.")
