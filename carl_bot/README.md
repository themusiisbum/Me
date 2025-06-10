# Carl Chatbot

This is a simple AI chatbot named **Carl**. It uses the
[Hugging Face Transformers](https://huggingface.co/transformers/) library to
generate responses using a text-generation model such as `gpt2`.

## Requirements

- Python 3.8+
- `transformers` library
- `torch` (required by transformers)

Install dependencies with:

```bash
pip install transformers torch
```

Optionally set the `CARL_MODEL` environment variable to choose a different model.

## Usage

Run the chatbot from the command line:

```bash
python carl_chatbot.py
```

Type your messages after the `You:` prompt. Type `exit`, `quit`, or `bye` to
leave the chat.
