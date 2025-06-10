import os
from transformers import pipeline


def load_model():
    model_name = os.environ.get("CARL_MODEL", "gpt2")
    generator = pipeline("text-generation", model=model_name)
    return generator

def generate_reply(generator, prompt, max_length=50):
    response = generator(prompt, max_length=max_length, num_return_sequences=1)
    text = response[0]['generated_text']
    # Remove the prompt from the generated text if repeated
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text.strip()

def chat():
    print("Carl: Hello! I'm Carl, your friendly AI chatbot. Ask me anything!")
    generator = load_model()
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Carl: Goodbye!")
            break
        prompt = f"User: {user_input}\nCarl:" 
        reply = generate_reply(generator, prompt)
        print(f"Carl: {reply}")

if __name__ == "__main__":
    chat()
