import streamlit as st

def render_footer():
    st.markdown('''
    <footer class="footer-bg" id="contato">
        <div class="footer container">
        <div class="footer-informacoes">
            <h3 class="font-2-l-b cor-0">Mapa do site</h3>
            <nav>
            <ul class="font-1-m cor-5">
                <li><a href="">Sobre o projeto</a></li>
                <li><a href="">Processar imagem</a></li>
                <li><a href="">Quem somos</a></li>
            </ul>
            </nav>
        </div>
        <div class="footer-contato">
            <h3 class="font-2-l-b cor-0">Contato</h3>
            <ul>
            <li>
                59.625-900, R. Francisco Mota, 572 - <p>Pres. Costa e Silva, Mossoró - RN</p>
            </li>
            <li><a href="">+55 (85) 9 9797-6640</a></li>
            <li><a href="">alunos@ufersa.edu.br</a></li>
            </ul>
        </div>
        </div>
        <p class="dir01">© Visão Tráfego 2023 • Todos os direitos reservados.</p>
    </footer>
    ''', unsafe_allow_html=True)