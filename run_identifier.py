import os
from ssd import ssd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    running = True
    while running:
        os.system('clear')

        print("************************************************")
        print("************** DETECTOR DE OBEJTOS *************")
        print("************************************************")
        print("\n\nEscolha qual arquitetura você deseja utilizar para detectar objetos na sua imagem...")
        print("\n1 - YOLO")
        print("\n2 - CNN")
        print("\n3 - SSD")
        print("0 - Sair e desabilitar lembretes")

        option = input("\nEscolha uma opção:")

        if option == '0':
            print("Até mais!")
            running = False
            break

        elif option in ['1', '2', '3']:
            img_path = input("\nAgora digite o caminho da imagem que irá utilizar:")
            ssd.local_identify_objects(img_path_file=img_path, output_dir=CURRENT_DIR)
            pass

        else:
            print("Opção inválida!")

        input("\nPressione qualquer tecla para voltar ao menu principal...")

        # Clear terminal
        if(os.name == 'posix'):
            os.system('clear')
        else:
            os.system('cls')


    st.write('Corpo do site!')

if __name__ == "__main__":
    # train yolo
    # train ssd
    # train cnn
    main()