import streamlit as st

def render_introducao():
    st.markdown('''
    <main class="introducao-bg" id="sobreProjeto">
        <div class="introducao container">
        <div class="intro-container">
            <div class="teste">
            <div class="frame45">
                <span class="text-introducao">
                <span class="text1">Projeto</span></br>
                <h4>Visão Tráfego</h4>
                </span>
                <span class="text3">
                <span>
                    Este projeto é capaz de identificar pedestres, carros, motos,
                    caminhões e bicicletas através do processamento digital de
                    imagens e aprendizado de máquina. Abaixo, você encontra um
                    formulário para testar o potencial do Visão Tráfego.
                </span>
                </span>
            </div>
            </div>
    </main>
    ''', unsafe_allow_html=True)