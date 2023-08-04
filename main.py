import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel

def get_input_text():
    return input_entry.text()

def display_output_text():
    output_entry.setText(get_input_text())

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("GUI 示例")

# 创建布局
layout = QVBoxLayout()

# 创建输入文本框
input_entry = QLineEdit()
layout.addWidget(input_entry)

# 创建输出文本框
output_entry = QLineEdit()
output_entry.setReadOnly(True)
layout.addWidget(output_entry)

# 创建提交按钮
submit_button = QPushButton("提交")
submit_button.clicked.connect(display_output_text)
layout.addWidget(submit_button)

# 设置布局
window.setLayout(layout)
window.show()

sys.exit(app.exec_())
