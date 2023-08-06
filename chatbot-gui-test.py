import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from transformers import AutoTokenizer, AutoModelForCausalLM
import contextlib

class ChatGPTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium', padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')
        
        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)

        self.input_edit = QLineEdit()
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.generate_response)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.submit_button)

        layout = QVBoxLayout()
        layout.addWidget(self.output_edit)
        layout.addLayout(input_layout)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

        self.setWindowTitle("Chatbot-GUI")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f2f2f2;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 14px;
                padding: 5px;
            }
            QLineEdit {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 5px;
                font-size: 14px;
                padding: 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.show()

    def generate_response(self):
        user_input = self.input_edit.text()
        if user_input.strip().lower() in ['exit', 'quit', 'bye']:
            self.output_edit.append("Chatbot: Goodbye!")
            self.close()

        input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        
        # Redirect both stdout and stderr to suppress warnings
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            chatbot_output = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        
        chatbot_response = self.tokenizer.decode(chatbot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        self.output_edit.append(f"User: {user_input}")
        self.output_edit.append(f"Chatbot: {chatbot_response}")
        self.input_edit.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatGPTApp()
    sys.exit(app.exec_())
