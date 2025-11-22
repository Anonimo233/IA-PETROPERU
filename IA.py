los datos cada vez que la aplicación se recimport streamlit as st
import pandas as pd
import numpyarga (lo que as np
import ocurre a menudo en Streamlit).

Aquí tienes el código inicial. Necesitarás `pandas random
import datetime # Para simular datos a lo largo del tiempo

# --- CONFIGURACIÓN DE LA PÁGINA ---`, `streamlit` y `numpy` (
st.set_page_configsi aún no lo tienes):(page_title="Monitoreo Financiero Petroper
`pip install streamlitú (Simulado)", page_icon="📈", layout="wide pandas numpy`

```python
import streamlit as st
import pandas as pd
import numpy as np
import")

st.title("📈 Monitoreo Financiero de Petroperú (Simulado con IA Conceptual)")

st.markdown(
    """
    Esta random
import time # Para simular el tiempo real
import matplotlib.pyplot as plt
import seaborn as sns

 aplicación simula un **panel de monitoreo en tiempo real** para el área financiera# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set de Petroperú.
    Muestra métricas clave_page_config(page_title="Monitoreo Financiero Petroper, tendencias y detección de anomalías (simuladas),ú (Simulado)", page_icon="📈", layout="wide")

st.title("📈 Monitoreo Financiero Petro
    además de conceptualizar cómo la **Inteligencia Artificial** potenciaría un sistema real.

perú (Simulado)")

st.markdown(
    """
    Esta aplicación simula un **dashboard de monitoreo financiero    **Disclaimer:** Todos los datos son generados aleatoriamente y no representan información real de Petroperú.
    """ en tiempo real**
)
st para Petroperú,.write("---")

# ---
    mostrando métricas clave GENERACIÓN DE DATOS SIMULADOS ---, tendencias y un panel

# R conceptual para la detección de anomalíasango de fechas para los datos históricos
start_date
    impulsada por Inteligencia Artificial.
    """
)
st.write("---")

 = datetime.date(2023, 1, 1)# --- FUNCIÓN PARA GENERAR DATOS SIMULADOS ---
# Usaremos st.cache_
end_date = datetime.date.today()
delta = enddata para que los datos no cambien en cada interacción
# pero_date - start_date

dates = [start_date + datetime.timedelta( la "actualización en tiempo real" la manejaremos condays=i) for i in range(delta.days + 1)]

# Simulación de Precios del Petróleo (Brent)
precio una lógica de refresco
_brent = [70 + @st.cache_data(ttl=5) # Los datos se refrescarán cada 5 segundos para simular "tiempo real"
def generate_20 * np.sin(i/30) + random.uniform(-5, 5)financial_data():
    current_time = pd.Timestamp.now()
    dates = pd.date_range(end for i in range(len(dates))]
# Simulación de Tasa de Cambio (USD a PEN)
tipo_cambio = [3.7=current_time, periods=2 + 0.1 * np.cos4, freq='H') # Últimas 24 horas

    # Simulación de precios del petróleo (Brent)
    (i/45) + random.uniform(-0.05, 0.05) for i in range(len(dates))]
# Simulación# Partimos de un precio base y añad de Ingresos Diarios (en millones deimos ruido y una ligera tendencia
    base USD)
ingresos_diarios = [150 + 50 * np.sin(i/2_price = 805) + random.uniform(-20, 20) for i in range(len(
    oil_prices = [base_price + random.uniform(-5, 5)dates))]
# Simulación de Gast for _ in range(len(dates))]
    oil_prices = [max(70, min(90, pos Operativos Diarios (en millones de USD)
gastos_diarios = [90 + 30 * np.cos(i/2)) for p in oil_prices] # Limitar0) + random.uniform(-15, 15) for i in range(len(dates))]


 entre 70 y 90

    # Simulación# Crear DataFrame principal de Ingresos y Gastos (millones de USD)
    # Los ingresos y gastos pueden fluctu
df_financiero = pd.DataFrame({
    "Fecha": dates,
    "Precio Brent (USD/barar y tener una relación con el precio del petróleo
    revenues = [pril)": precio_brent,
    "Tipo de Cambio (USD/PEN)": tipo_cambio * random.uniform(1.2, 1.5) *,
    "Ingresos Diarios (MM USD)": ingresos random.uniform(50, 100) / 1000 for p in oil_prices]
    expenses = [p * random._diarios,
    "Gastos Diarios (MM USD)": gastos_diarios
})

# Asegurar que los datos financieros no sean negativos
df_financiero["uniform(0.8, 1.1) * random.uniform(50, 1Ingresos Diarios (MM USD)"] = df_financiero["Ingresos Diarios (MM USD)"].apply00) / 1000 for p in oil_prices]

    # Añ(lambda x: max(x, 10))
df_financiero["Gastadir una anomalía simulada en un puntoos Diarios (MM USD)"] = df_financiero["Gastos Diarios (MM aleatorio
    if random.random() < 0 USD)"].apply(lambda x: max(x, 5.3: # 30% de probabilidad de anomalía
        anomaly))

df_financiero["Margen Diario_idx = random.randint(5, len(dates (MM USD)"] = df_financiero["Ingresos Diarios () - 2)
        if random.random() < 0.5: # Anomalía positiva (MM USD)"] - df_financiero["Gastos Diarios (MM USD)"]

# --- HOej. pico de ingresos inesperado)
            revenues[anomaly_Y: Últimos datos simulidx] *= random.uniform(1.5, ados ---
today_data = df_financiero.iloc[-1]2.5)
            st.session_state.anomaly
yesterday_data = df_financiero.iloc_alert = f"ALERTA: Pico in[-2]

# --- DASHBOARD DE MÉTRICAS CLusual de ingresos en {dates[anomaly_AVE ---
st.header("📊 Métricas Financieras Clidx].strftime('%H:%M')}!"
        elseave (Hoy)")

col1, col2, col3, col4: # Anomalía negativa (ej. caída de ingresos = st.columns(4)

 o pico de gastos)
            revenues[anomaly_idx]with col1:
    st.metric(
        label="Precio Brent *= random.uniform(0.5, 0.7)
            expenses[anomaly_idx] *= random Actual",
        value=f"{today_data['Precio Brent (USD/barril)']:..uniform(1.5, 2.0)
            st.session_2f} USD",
        delta=f"{today_data['Precio Brentstate.anomaly_alert = f"ALERTA: Des (USD/barril)'] - yesterday_data['Precio Brent (USD/viación significativa de fondos/ingresos enbarril)']:.2f} USD"
    )
    st.caption("Precio {dates[anomaly_idx].strftime('%H:%M')}!"
    else:
        st del petróleo crudo de referencia.")
with col2:
    st.session_state.anomaly_alert = "Monitoreo normal. No se detectaron anomalías.".metric(
        label="Tipo de Cambio Actual",
        value=f"{today_data['Tipo de Cambio (USD/


    df = pd.DataFrame({
PEN)']:.3f} PEN",
        delta=f"{today_        "Fecha": dates,
        data['Tipo de Cambio (USD/"Precio Petróleo Brent (USD/barril)": oil_prices,
        "Ingresos (MMPEN)'] - yesterday_data['Tipo de Cambio (USD/PEN)']:.3f} PEN"
    )
    st. USD)": revenues,
        "Gastos (MM USD)": expenses
caption("USD a Soles Peruanos.")
with col3:    })
    df["Beneficio (MM USD)
    st.metric(
        label="Ingresos"] = df["Ingresos (MM Diarios",
        value=f"{today_data['Ingresos Diarios (MM USD)']:.1 USD)"] - df["Gastos (MM USD)"]
    dff} MM USD",
        delta=f"{today_data['Ingresos Diarios ( = df.set_index("Fecha")
    return df

# Inicializar estado de la sesión si no existe
if 'MM USD)'] - yesterday_data['Ingresos Diarios (MM USD)']:.1f} MM USD"
anomaly_alert' not in st.session_state:    )
    st.caption("Estim
    st.session_state.anomaly_alert =ación de ingresos de hoy.")
with col4:
    st.metric(
        label=" "Iniciando monitoreo..."

# Obtener los datos (se refrescarán cada 5 segundos debido alMargen Operativo Diario",
        value=f"{today_data ttl de st.cache_data)
financial_df = generate_financial['Margen Diario (MM USD)']:.1f} MM USD",
        delta=f_data()

# --- VISIÓN GENERAL: MÉTRICAS CLAVE ---
st"{today_data['Margen Diario (MM USD)'] - yesterday_data['Margen Diario (MM USD)']:.1f}.header("📊 Visión General del Desempeño Financ MM USD",
        delta_color="normal" # Positivo esiero")

col1, col2, col3, col4 = st bueno
    )
    st.caption("Ganancia antes de costos no operativos.columns(4)

with col1:
    st.metric(.")

st.write("---")

# --- TENDENCIAS Y PRONÓSTICOS (Visuallabel="Ingresos (última hora)",
ización Simple) ---
st.header("📈              value=f"{financial_df['Ingresos (MM USD)'].iloc[-1]:. Tendencias y Análisis Histórico")

st.subheader("Precio2f} MM USD")
    st.write(f"Promedio  del Petróleo Brent (Últimos 324h: {financial_df['Ingresos (MM USD)'].0 días)")
# Filtramos para mostrar solo los últimosmean():.2f} MM USD")

with col2:
    st.metric(label="Gastos (última hora 30 días para una mejor visualización de tendencia
df_last_30_days =)",
              value=f"{financial_df['Gastos (MM USD)'].iloc[- df_financiero.tail(30).set_index("1]:.2f} MM USDFecha")
st.line_chart(df_last_3")
    st.write(f0_days["Precio Brent (USD/barril)"Promedio 24h: {financial_df['Gastos (MM USD)'].mean():.2"], use_container_width=Truef} MM USD")

with col3:
    )
st.caption("Visualización de la evolución diariast.metric(label="Beneficio (última hora)",
              value=f"{financial_df['Beneficio (MM USD del precio del Brent. Una IA predict)'].iloc[-1]:.2f} MM USDiva podría pronosticar su valor futuro.")

st.subheader("Ingresos y Gastos",
              delta=f" Diarios (Últimos 30 días)")
st.line_chart(df{(financial_df['Beneficio (MM USD)'].iloc[-1] - financial_df['Beneficio (_last_30_days[["Ingresos Diarios (MM USD)",MM USD)'].iloc[-2]):.2f} MM USD vs. hora anterior")
    st. "Gastos Diarios (MM USD)"]], use_container_width=True)
write(f"Promedio 24h: {financial_df['st.caption("Comparativa de los flBeneficio (MM USD)'].mean():.2f} MM USD")ujos de entrada y salida. Una IA podría optimizar la gestión de gastos o pre

with col4:
    st.metric(label="Precio Brent (última hora)",
ver desviaciones en ingresos.")

st.write("---              value=f")

# --- DETECCIÓN DE"{financial_df['Precio Petróleo Brent (USD/barril)'].iloc[-1]: ANOMALÍAS (Simulada) ---.2f} USD")
    st.write(f"Promedio 24h:
st.header("🚨 Detección de Anomalías (Simulada)")

# Simular una anomalía inyectando un valor in {financial_df['Precio Petróleo Brent (USD/usual en un día reciente
anom_barril)'].mean():.2f} USD")

date_idx = -random.randint(5st.write("---")

# --- TENDENCIAS CLAVE ---, 15) # Un día al azar en los últimos 5-15 días
st.header("📈 Tendencias Financieras (Últimas 24 Horas)")

# Gráfico
df_financ de Ingresos yiero.loc[df_financiero.index[anom_date_idx], "Ing Gastos
fig_fin, ax_fin = plt.subplots(figsize=(12resos Diarios (MM USD)"] *= random, 5))
ax_fin.plot.uniform(0.3, 0.5) # Baja dr(financial_df.index, financial_df["Ingresos (MM USD)"], labelástica
df_financiero.loc[df_financiero.index[anom_date="Ingresos", color="green")
ax_fin.plot(financial_df._idx], "Gastos Diarios (MM USD)"] *= random.uniform(1.index, financial_df["Gastos (MM USD)8, 2.5) # Sube drástica

st.markdown"], label="Gastos", color="red")
ax_fin.set_title("Ingresos y(f"**Alerta Simulada:** ¡Pos Gastos (MM USD)")
ax_fin.set_xlabel("Horaible anomalía detectada en los datos financieros recientes!")")
ax_fin.set_ylabel("M

# Resaltar la anomalía enonto (MM USD)")
ax_fin.tick_params(axis='x', rotation un gráfico
st.subheader("Margen Oper=45)
ax_fin.legend()
plt.tight_ativo Diario con Anomalía")
df_plot_anom = df_financiero.taillayout()
st.pyplot(fig_fin)
(60).set_index("Fecha")plt.close(fig_fin) # Importante para liberar memoria

# # Mirar los últimos 60 Gráfico de Precio del Petróleo
fig_oil días

# Crear una columna para resaltar la anomalía
, ax_oil = plt.subplots(figsize=(12, 5df_plot_anom['Anomalía'] = None))
ax_oil.plot(
if df_financiero.index[anom_financial_df.index, financial_df["Precio Petróleo Brent (USD/barril)"], label="Preciodate_idx] in df_plot_anom.index:
    df_plot_ Brent", color="blue")
ax_oil.setanom.loc[df_financiero.index[anom_date_idx], '_title("Precio Petróleo Brent (USD/barril)")
ax_oil.set_xlabel("Hora")Anomalía'] = df_plot_anom.loc[df_financiero.index
ax_oil.set_ylabel("Precio (USD[anom_date_idx], 'Margen Diario (MM USD)']

# Dibujar con)")
ax_oil.tick_params(axis='x', rotation=45)
ax_oil.legend()
plt st.line_chart y una columna de puntos para la anomalía
st.line_.tight_layout()
st.pyplot(fig_oil)
plt.close(fig_oil) #chart(df_plot_anom[['Margen Diario ( Importante para liberar memoria

st.write("---")

# --- DETECCIÓN DE ANMM USD)', 'Anomalía']], use_container_width=True)
st.markdownOMALÍAS (SIMULADA POR IA) ---
(
    f"""
    <div style='st.header("🚨 Detección de Anomalbackground-color: #fff3cd; color: #856ías (AI Simulada)")
st.warning(st.session_state.anomaly_alert404)
st.markdown(
    """
    *Aquí, un sistema de IA real analizaría patrones históricos; padding: 10px; border-radius: 5px;'>
        La IA de detección de anomalías y ha marcado en tiempo real para identificar
    desviaciones significativas en los datos financieros que una posible desviación significativa en el margen operativo del puedan indicar fraude, errores operativos,
    cambios bruscos en el
        **{df_financiero.loc[df_financiero.index[anom_date_idx], ' mercado o eventos inesperados. Los modelos podrían incluir:*Fecha'].strftime('%Y-%m-%d')}
    -**. **Algoritmos de Series Temporales:** ARIMA,
        Se recomienda revisión urgente.
    </div>
 Prophet, Holt-Winters para predecir valores esperados.
    - **    """,
    unsafe_allow_html=True
)

st.write("---Modelos de Aprend")

# --- CONCEPTO DEizaje No Supervisado:** K-Means, Isolation Forest, IA EN EL MONITOREO FINANCIERO ---
st.header("🧠 One-Class SVM para detectar outliers Cómo una IA Real Mejoraría este Monitoreo").
    - **Redes Neuronales Recurrentes (RNN

st.markdown(
    """
    Un sistema des):** Para aprender IA real ir patrones complejos en secuencias de datos.
    """
)
st.write("---")ía mucho más allá de las visualizaciones básicas, integrando modelos

# --- ANÁLISIS DE RIESGO Y PROYECC de Machine Learning avanzados para:

    1.  **Predicción de PreIONES (CONCEPTUAL) ---
st.header("🔮 Análisis de Riesgocios y Volatilidad:**
        *   Modelos de Series Temporales (AR y Proyecciones (Conceptos de IA)")
st.infoIMA, Prophet, LSTM) para pronosticar precios de petróleo, tipo de("Esta sección es conceptual y muestra dónde cambio y demanda de productos con alta precisión.
 la IA podría proporcionar análisis más profundos.")
st.markdown(
        *   Análisis de factores macroeconómicos,    """
    Un módulo de IA más avanzado podría ofrecer: geopolíticos y patrones históricos.
        *   **Beneficio:** Mej
    *ora la toma de decisiones de cobertura,   **Proyecciones Financieras:** Pronósticos de ingresos, gastos compra/venta y planificación estratégica.
    2.   y beneficio para los próximos días/semanas, basados en precios futuros**Detección de Fraude y Anomalías:**
        *    de commodities,Algoritmos de detección capacidad de producción y demanda.
    *   **Evalu de valores atípicos (Isolation Forest, One-Class SVM)ación de Riesgos:** Identificar y cuantificar riesgos asociados a fluctuaciones del tipo de cambio, para identificar transacciones inusuales, patrones de gastos anómalos o desviaciones inesperadas en ingresos.
        *   Monitoreo de auditorías y concili volatilidad del precio del petróleo, interrupciones en la cadena de suministro o cambios regulatoraciones automáticas.
        *   **Benefios.
    *   **Optimización de Opericio:** Minimiza riesgos financieros, previene pérdidasaciones:** Sugerencias para optimizar la compra/ por fraude y errores operativos.
    3venta de petróleo, gestión de inventarios y asignación de.  **Análisis de Riesgo Cuantitativo:**
        *   Model recursos.
    *   **Análisis de Sentimientosos de riesgo de crédito, riesgo de mercado y riesgo operativo basados en datos históricos:** Monitorear noticias y redes sociales para evaluar el impacto en la reput y factores externos.
        *   Simación y las finulaciones Monte Carlo para evaluar el impacto de diferentesanzas de Petroperú.
    """
)
st.write("---")
st.markdown(" escenarios.
        *   **BenefDashboard de Monitoreo Financiero deicio:** Permite una gestión de riesgo proactiva y una asignación de capital más eficiente.
    4. Petroperú - **Solo con fines demostrativos y simulados.**")

  **Optimización de Operaciones y Log```

### Cómo ejecutar este código:

1.  Aística:**
        *   Modelos de optimización para lasegúrate de tener instaladas las cadena de suministro, refinación y distribución, considerando costos librerías: `pip install streamlit pandas numpy, demanda y capacidad.
        * matplotlib seaborn`
2.  Guarda el código como   **Beneficio:** Reduce costos operativos, mejora `petroperu_monitor.py` (o similar).
3. la eficiencia y maximiza la rentabilidad.
    5.  **Análisis de Sentimientos de Noticias Financ  Ejecútalo desde tu terminal: `streamlit run petroperuieras:**
        *   Procesamiento de Leng_monitor.py`

### Notas Importuaje Natural (NLP) para analizar noticias, informes y redes sociales, identificando eventosantes:

*   **Simulación de Tiempo Real:** El decor que podrían afectar los mercados oador `@st.cache_data( la reputación de la empresa.
        *   **Beneficio:** Alttl=5)` en `generate_financial_data()` hará que Streamertas tempranas sobrelit regenere los datos cada 5 segundos. Verás cómo los números y gráficos eventos críticos y comprensión del entorno se actualizan automáticamente, de mercado.

    La IA transformaría este panel en simulando un flujo de datos en una herramienta inteligente capaz vivo.
*   **Anomalías Simuladas:** Hay una probabilidad del 30% de que aparezca una "alerta de anomalía" cada vez que los datos se refrescan de ofrecer, con picos o caídas inesperadas en los ingresos insights predictivos, alert/gastos.
*   **Conceptos de IA:** Lasar proactivamente sobre riesgos y optimizar las operaciones financieras de Petroperú.
    """
)

st.write("---")
st.markdown secciones de "D("Desarrollado con 🚀 en Python y Streamlit para fines demostetección de Anomalías" y "Análisis de Riesgo" están diseñadas para explicar qué tipo de modelos de IA y funcionalidadesrativos.")

# Para que el modelo de imagen no falle
