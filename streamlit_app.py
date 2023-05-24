import streamlit as st

uploaded_file = st.file_uploader("Escolher Imagem")
option = st.selectbox(
    'Qual arquitetura será utilizada na identificação?',
    ('YOLO', 'CNN', 'SSD'))

st.write('Identificar objetos usando:', option)
