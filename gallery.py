import streamlit as st

def render_gallery():

  col1, col2, col3, col4, col5, col6 = st.columns(6)

  with col1:
    st.image("front-end/img/allef.png", caption='Áleff Jonathan')

  with col2:
    st.image("front-end/img/dyego.png", caption='Dyego Magno')

  with col3:
    st.image("front-end/img/erlon.png", caption='Erlon Dantas')

  with col4:
    st.image("front-end/img/felipe.png", caption='Felipe Gabriel')

  with col5:
    st.image("front-end/img/gidel.png", caption='José Gidel')

  with col6:
    st.image("front-end/img/mateus.png", caption='José Mateus')


  col1, col2, col3, col4, col5 = st.columns(5)

  with col1:
    st.image("front-end/img/karin.png", caption='Karin de Fátima')

  with col2:
    st.image("front-end/img/leandro.png", caption='Leandro Gameleira')

  with col3:
    st.image("front-end/img/ryllari.png", caption='Ryllari Raianne')

  with col4:
    st.image("front-end/img/victor.png", caption='Victor Benoiston')

  with col5:
    st.image("front-end/img/willian.png", caption='William Donizete')



