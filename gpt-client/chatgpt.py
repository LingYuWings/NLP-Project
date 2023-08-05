# chatgpt.py

import openai

class ChatGPTClient:
    def __init__(self, api_key):
        openai.api_key = api_key

    def get_chat_response(self, message):
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=message,
            max_tokens=150
        )

        if "choices" in response and response["choices"]:
            return response["choices"][0]["text"].strip()
        else:
            raise ValueError("Invalid API response. 'choices' key not found or empty.")
