# main.py

import sys
from PyQt5.QtWidgets import QApplication
from gui import ChatGPTClientGUI
from chatgpt import ChatGPTClient

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(open("style.qss").read())

    # Initialize the ChatGPTClient with an empty API key
    chatgpt_client = ChatGPTClient("")

    window = ChatGPTClientGUI()
    window.set_chatgpt_client(chatgpt_client)
    window.show()

    sys.exit(app.exec_())
