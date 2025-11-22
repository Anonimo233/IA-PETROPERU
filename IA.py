import streamlit as st
import pandas as pd
import numpy as np

# 1. Configuración de la página
st.title('🤖 Mi Primera IA: Predicción de Salarios')
st.write('Esta IA aprende de datos simples para predecir salarios según años de experiencia.')

# 2. Datos de entrenamiento (Dataset simulado)
# En la vida real, cargarías esto desde un CSV/Excel
data = {
    'Años_Experiencia': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salario_Anual': [30000, 35000, 42000, 48000, 55000, 65000, 72000, 80000, 95000, 110000]
}
df = pd.DataFrame(data)

# 3. Entrenar el modelo (Machine Learning)
# Separamos las características (X) de lo que queremos predecir (y)
X = df[['Años_Experiencia']]
y = df['Salario_Anual']

model = LinearRegression()
model.fit(X, y) # Aquí ocurre la "magia" del aprendizaje

# 4. Interfaz de Usuario en Streamlit
st.sidebar.header("Parámetros")
input_experiencia = st.sidebar.slider('Selecciona años de experiencia:', 0, 15, 5)

# 5. Hacer la predicción
# El modelo recibe el valor del slider y calcula el resultado
prediccion = model.predict([[input_experiencia]])

# 6. Mostrar resultados
st.subheader('Resultados:')
st.metric(label="Salario Estimado", value=f"${prediccion[0]:,.2f} USD")

# Bonus: Mostrar gráfico
st.subheader('Visualización del Modelo')
chart_data = df.copy()
# Añadimos la predicción actual al gráfico para ver dónde cae
st.line_chart(chart_data.set_index('Años_Experiencia'))
