import streamlit as st

def render_navbar():
    st.markdown('''
    <header class="header-bg">
        <nav class="header">
        <ul class="header-menu">
            <div></div>
            <li><a href="#sobreProjeto" class="text">Sobre o projeto</a></li>
            <li><a href="#processarImagem" class="text">Processar imagem</a></li>
            <li><a href="#quemSomos" class="text">Quem somos</a></li>
            <li><a href="#contato" class="text"> Contato </a></li>
        </ul>
        </nav>
    </header>
    ''', unsafe_allow_html=True)