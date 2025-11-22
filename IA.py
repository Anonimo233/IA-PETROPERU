import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="PetroDashboard AI", layout="wide", page_icon="🛢️")

# TÍTULO Y CONTEXTO
st.title("🛢️ Petro-Monitor AI: Dashboard Financiero en Tiempo Real")
st.markdown("""
Esta IA monitorea variables macroeconómicas críticas para la salud financiera de una petrolera
y utiliza un algoritmo de reglas para determinar el **Nivel de Riesgo Operativo**.
""")

# BARRA LATERAL
st.sidebar.header("Configuración de IA")
umbral_petroleo = st.sidebar.slider("Alerta si Petróleo cae bajo ($):", 50, 100, 70)
frecuencia = st.sidebar.selectbox("Frecuencia de actualización", ["1 min", "5 min", "1 hora"])

# --- FUNCIÓN PARA OBTENER DATOS EN VIVO ---
def obtener_datos_mercado():
    # CL=F es el futuro del Petróleo Crudo (WTI)
    # PEN=X es el tipo de cambio Soles vs Dólares
    tickers = yf.tickers.Tickers("CL=F PEN=X")
    
    # Descargamos datos del día
    petroleo = tickers.tickers['CL=F'].history(period="1d")
    tipo_cambio = tickers.tickers['PEN=X'].history(period="1d")
    
    precio_oil = petroleo['Close'].iloc[-1]
    precio_dolar = tipo_cambio['Close'].iloc[-1]
    
    # Variación porcentual respecto a la apertura
    var_oil = ((precio_oil - petroleo['Open'].iloc[-1]) / petroleo['Open'].iloc[-1]) * 100
    
    return precio_oil, precio_dolar, var_oil

# --- LÓGICA DE IA (MOTOR DE RIESGO) ---
def analizar_riesgo(precio_oil, precio_dolar, umbral):
    score = 0
    razon = []
    
    # Regla 1: Precio del crudo (Ingresos)
    if precio_oil < umbral:
        score += 50
        razon.append(f"⚠️ Precio del crudo bajo (${precio_oil:.2f}) reduce márgenes.")
    else:
        razon.append("✅ Precio del crudo saludable.")
        
    # Regla 2: Tipo de cambio (Deuda)
    # Si el dólar sube mucho, la deuda en dólares duele más
    if precio_dolar > 3.80:
        score += 30
        razon.append(f"⚠️ Dólar alto (S/{precio_dolar:.2f}) impacta importación y deuda.")
    
    # Determinación
    if score >= 50:
        estado = "ALERTA ROJA"
        color = "red"
    elif score >= 30:
        estado = "PRECAUCIÓN"
        color = "orange"
    else:
        estado = "ESTABLE"
        color = "green"
        
    return estado, color, razon

# --- EJECUCIÓN DEL DASHBOARD ---
try:
    # 1. Obtener datos
    with st.spinner('Conectando con mercados internacionales...'):
        oil, dolar, var_oil = obtener_datos_mercado()
    
    # 2. Análisis de IA
    estado, color, razones = analizar_riesgo(oil, dolar, umbral_petroleo)

    # 3. Métricas Principales (KPIs)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Petróleo WTI (Ref. Ingresos)", f"${oil:.2f}", f"{var_oil:.2f}%")
    
    with col2:
        st.metric("Dólar (Ref. Costos)", f"S/ {dolar:.2f}")
        
    with col3:
        st.markdown(f"### Estado Financiero: :{color}[{estado}]")

    # 4. Explicación de la IA
    st.info("🤖 **Análisis de la IA:**")
    for r in razones:
        st.write(f"- {r}")

    # 5. Gráfico de Tendencia (Simulado para el ejemplo visual)
    st.subheader("Tendencia de Volatilidad (Simulación 30 días)")
    # Generamos datos dummy para visualizar cómo se vería el gráfico histórico
    fechas = pd.date_range(start='2023-01-01', periods=30)
    datos_simulados = pd.DataFrame({
        'Fecha': fechas,
        'Cash Flow Proyectado': [oil * 1000 * (1 + i*0.01) for i in range(30)]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos_simulados['Fecha'], y=datos_simulados['Cash Flow Proyectado'],
                        mode='lines+markers', name='Flujo de Caja'))
    st.plotly_chart(fig, use_container_width=True)

    # Botón de actualización manual
    if st.button('Actualizar Análisis'):
        st.rerun()

except Exception as e:
    st.error(f"Error al conectar con los datos financieros: {e}")
    st.warning("Nota: Los mercados pueden estar cerrados en este momento.")

