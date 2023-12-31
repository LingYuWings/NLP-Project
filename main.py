import sys
import openai
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import QFile, QTextStream, Qt

# 设置你的OpenAI API密钥
openai.api_key = "YOUR_OPENAI_API_KEY"

def load_stylesheet(file_path):
    style_file = QFile(file_path)
    if style_file.open(QFile.ReadOnly | QFile.Text):
        stream = QTextStream(style_file)
        return stream.readAll()
    return ""

def send_message_to_chat(message):
    response = openai.Completion.create(
        engine="davinci",  # 选择OpenAI引擎
        prompt=message,   # 输入内容
        max_tokens=100,    # 最大生成的令牌数量
        stop=None          # 停止符号，不设置停止符号，生成完整的回复
    )
    return response.choices[0].text.strip()

def get_input_text():
    return input_entry.text()

def display_output_text():
    input_text = get_input_text()
    output_text = send_message_to_chat(input_text)
    output_entry.setText(output_text)

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Chat GPT 示例")

# 加载样式表
style = load_stylesheet("style.qss")
app.setStyleSheet(style)

# 创建网格布局
grid_layout = QGridLayout()

# 创建输入文本框
input_entry = QLineEdit()
input_entry.setFixedWidth(500)  # 设置输入文本框宽度为 200 像素
input_entry.setFixedHeight(300)
input_entry.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 将文本内容从左上角开始
grid_layout.addWidget(input_entry, 0, 0, 1, 2)

# 创建提交按钮
submit_button = QPushButton("提交")
submit_button.setFixedWidth(100)  # 设置提交按钮宽度为 100 像素
submit_button.clicked.connect(display_output_text)
grid_layout.addWidget(submit_button, 1, 1)

# 创建输出文本框
output_entry = QLineEdit()
output_entry.setReadOnly(True)
output_entry.setFixedWidth(500)  # 设置输出文本框宽度为 200 像素
output_entry.setFixedHeight(300)
output_entry.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 将文本内容从左上角开始
grid_layout.addWidget(output_entry, 2, 0, 1, 2)

# 设置布局
window.setLayout(grid_layout)
window.show()

sys.exit(app.exec_())