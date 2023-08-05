import sys
import openai
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLineEdit, QPushButton, QLabel, QComboBox
from PyQt5.QtCore import QFile, QTextStream, Qt

# 设置OpenAI API密钥
openai.api_key = None

prompts = {
    "CN to JP": "./prompts/CN_to_JP",
    "CN to EN": "./prompts/CN_to_EN",
    "EN to JP": "./prompts/EN_to_JP"
}

def load_stylesheet(file_path):
    style_file = QFile(file_path)
    if style_file.open(QFile.ReadOnly | QFile.Text):
        stream = QTextStream(style_file)
        return stream.readAll()
    return ""

def send_message_to_chat(message, gpt_model, prompt):
    prompt_text = prompts.get(prompt, "")  # 获取用户选择的提示内容
    prompt_and_message = prompt_text + "\n" + message  # 将用户输入的内容与预设的提示进行拼接
    response = openai.Completion.create(
        engine=gpt_model,
        prompt=prompt_and_message,
        max_tokens=100,
        stop=None
    )
    return response.choices[0].text.strip()

def get_input_text():
    return input_entry.text()

def display_output_text():
    input_text = get_input_text()
    selected_gpt_model = gpt_model_combo.currentText()  # 获取选择的GPT模型
    output_text = send_message_to_chat(input_text, selected_gpt_model)
    output_entry.setText(output_text)
    
def update_api_key(key):
    openai.api_key = key

def update_api_key_text():
    api_key = api_key_entry.text()
    update_api_key(api_key)

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("GPT translation machine")

# 加载样式表
style = load_stylesheet("style.qss")
app.setStyleSheet(style)

# 创建网格布局
grid_layout = QGridLayout()


# 创建API密钥输入文本框
api_key_entry = QLineEdit()
api_key_entry.setFixedWidth(250)  # 设置输入文本框宽度为 200 像素
api_key_entry.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 将文本内容从左上角开始
grid_layout.addWidget(api_key_entry, 0, 0, 1, 2)

# 创建更新API密钥按钮
update_api_key_button = QPushButton("更新API密钥")
update_api_key_button.setFixedWidth(120)  # 设置提交按钮宽度为 100 像素
update_api_key_button.clicked.connect(update_api_key_text)
grid_layout.addWidget(update_api_key_button, 0, 1)

# 创建输入文本框
input_entry = QLineEdit()
input_entry.setFixedWidth(500)  # 设置输入文本框宽度为 200 像素
input_entry.setFixedHeight(300)
input_entry.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 将文本内容从左上角开始
grid_layout.addWidget(input_entry, 1, 0, 1, 2)

# 创建下拉选择框
gpt_model_combo = QComboBox()
gpt_model_combo.addItem("gpt-3.0")
gpt_model_combo.addItem("gpt-3.0-turbo")
gpt_model_combo.addItem("gpt-4.0")
gpt_model_combo.setFixedWidth(100)  # 设置下拉选择框宽度为 100 像素
grid_layout.addWidget(gpt_model_combo, 2, 0)

# 创建提交按钮
submit_button = QPushButton("提交")
submit_button.setFixedWidth(100)  # 设置提交按钮宽度为 100 像素
submit_button.clicked.connect(display_output_text)
grid_layout.addWidget(submit_button, 2, 1)

# 创建输出文本框
output_entry = QLineEdit()
output_entry.setReadOnly(True)
output_entry.setFixedWidth(500)  # 设置输出文本框宽度为 200 像素
output_entry.setFixedHeight(300)
output_entry.setAlignment(Qt.AlignTop | Qt.AlignLeft)  # 将文本内容从左上角开始
grid_layout.addWidget(output_entry, 3, 0, 1, 2)

# 创建下拉选择框（预设提示）
prompts_combo = QComboBox()
for prompt in prompts.keys():
    prompts_combo.addItem(prompt)
prompts_combo.setFixedWidth(150)  # 设置下拉选择框宽度为 150 像素
grid_layout.addWidget(prompts_combo, 4, 1)
# 设置布局
window.setLayout(grid_layout)
window.show()

sys.exit(app.exec_())