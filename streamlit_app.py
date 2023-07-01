import numpy as np
import streamlit as st
from ssd import ssd
from yolo import yolo
from navbar import render_navbar
from introduction import render_introduction
from information import render_information
from footer import render_footer
from gallery import render_gallery

with open('front-end/css/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

render_navbar()
render_introduction()


def prepare_params_data(id_params):
    params_data = {
        "class": [],
        "avg_conf": [],
        "count_objects": [],
    }
    for key, conf_list in id_params.items():
        params_data["class"].append(key)
        qntd = len(conf_list)
        if qntd > 0:
            params_data['avg_conf'].append(sum(conf_list)/qntd)
            params_data['count_objects'].append(qntd)
        else:
            params_data['avg_conf'].append("-")
            params_data['count_objects'].append("-")
    
    return params_data


input_form = st.form(key='input_form')
option = input_form.radio(
    'Escolha a arquitetura para a identificação:',
    ('YOLO', 'CNN', 'SSD'))
uploaded_file = input_form.file_uploader("Escolha a imagem que será identificada:", type=['png', 'jpg'])

submitted = input_form.form_submit_button("Submit")
if submitted:
    col1, col2 = st.columns(2)
    if option == 'SSD':
        id_image, id_params = ssd.identify_objects(uploaded_file)
        with col1:
            if uploaded_file:
                st.image(uploaded_file, caption='Imagem base')
        with col2:
            if id_image.any():
                st.image(id_image, caption='Imagem resultante')

        st.dataframe(
            prepare_params_data(id_params),
            column_config={
                "class": "Classe",
                "avg_conf": "Acurácia média",
                "count_objects": "Quantidade de objetos",
            },
            hide_index=True,
        )

    elif option == 'YOLO':
        id_image = yolo.identify_objects(uploaded_file)
        
        with col1:
            if uploaded_file:
                st.image(uploaded_file, caption='Imagem base')
        with col2:
            if id_image.any():
                st.image(id_image, caption='Imagem resultante')
        
    else:
        st.write('Opcao nao implementada ainda:', option)

render_information()
render_gallery()
render_footer()