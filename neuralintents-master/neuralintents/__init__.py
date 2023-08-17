from main import GenericAssistant

assistant = GenericAssistant('intents.json', model_name="test_model")
assistant.train_model()
assistant.save_model()

done = False

while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        assistant.request(message)