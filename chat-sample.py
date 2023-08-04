import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget
import openai

class ChatGPTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.text_edit = QTextEdit()
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.generate_response)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.send_button)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

        self.setWindowTitle("ChatGPT 应用")
        self.setStyleSheet("./chat.qss")

        self.show()

    def generate_response(self):
        user_input = self.text_edit.toPlainText()
        response = self.get_openai_response(user_input)
        self.text_edit.append(f"ChatGPT: {response}")

    def get_openai_response(self, input_text):
        # 与之前的代码保持一致
        openai.api_key = "你的OpenAI API密钥"  # 替换为你的实际API密钥
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=input_text,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatGPTApp()
    sys.exit(app.exec_())
