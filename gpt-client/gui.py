# gui.py

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QLineEdit, QSizePolicy
from PyQt5.QtCore import Qt
from chatgpt import ChatGPTClient

class ChatGPTClientGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ChatGPT Client")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        # Create the central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # API Key input and update button
        api_layout = QHBoxLayout()
        main_layout.addLayout(api_layout)

        self.api_key_input = QLineEdit(self)
        api_layout.addWidget(self.api_key_input)

        update_button = QPushButton("Update API Key", self)
        update_button.clicked.connect(self.update_api_key)
        api_layout.addWidget(update_button)

        # Chat display
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        main_layout.addWidget(self.chat_display)

        # Question input and submit button
        question_layout = QHBoxLayout()
        main_layout.addLayout(question_layout)

        self.question_input = QLineEdit(self)
        self.question_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        question_layout.addWidget(self.question_input)

        submit_button = QPushButton("Submit", self)
        submit_button.clicked.connect(self.submit_question)
        question_layout.addWidget(submit_button)

        # Initialize the ChatGPT client
        self.chatgpt_client = None

        # List to store chat history
        self.chat_history = []

    def set_chatgpt_client(self, api_key):
        self.chatgpt_client = ChatGPTClient(api_key)

    def update_api_key(self):
        api_key = self.api_key_input.text().strip()  # 去掉额外的空格
        self.chatgpt_client.api_key = api_key

    def submit_question(self):
        question = self.question_input.text()
        if not question or not self.chatgpt_client:
            return

        response = self.chatgpt_client.get_chat_response(question)

        # Append question and response to the chat history
        self.chat_history.append(("You: " + question, "ChatGPT: " + response))

        # Update the chat display with the updated chat history
        self.update_chat_display()

        # Clear the question input field
        self.question_input.clear()

    def update_chat_display(self):
        # Clear the chat display and update it with the chat history
        self.chat_display.clear()
        for chat in self.chat_history:
            self.chat_display.append(chat[0])
            self.chat_display.append(chat[1])
            self.chat_display.append("\n")
