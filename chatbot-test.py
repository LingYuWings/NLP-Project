from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import contextlib
import sys

# Load pre-trained DialoGPT model and tokenizer with padding_side='left'
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium', padding_side='left')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')

# Chatbot loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: Goodbye!")
        break

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Redirect both stdout and stderr to suppress warnings
    with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        chatbot_output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    chatbot_response = tokenizer.decode(chatbot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Chatbot:", chatbot_response)
