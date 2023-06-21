import numpy as np
import streamlit as st
from ssd import ssd

input_form = st.form(key='input_form')
option = input_form.radio(
    'Escolha a arquitetura para a identificação:',
    ('YOLO', 'CNN', 'SSD'))
uploaded_file = input_form.file_uploader("Escolha a imagem que será identificada:", type=['png', 'jpg'])

submitted = input_form.form_submit_button("Submit")
if submitted:
    col1, col2 = st.columns(2)
    if option == 'SSD':
        st.write('Identificar objetos usando:', option)
        id_image = ssd.identify_objects(uploaded_file)
        with col1:
            if uploaded_file:
                st.image(uploaded_file, caption='Imagem base')
        with col2:
            if id_image.any():
                st.image(id_image, caption='Imagem resultante')
            
    else:
        st.write('Opcao nao implementada ainda:', option)
