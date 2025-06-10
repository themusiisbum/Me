# Carl Chatbot

This is an advanced version of **Carl**, a chatbot implemented entirely with
Python's standard library.  It combines pattern matching, sentiment tracking
and a Markov chain text generator to create varied responses without relying
on external APIs or heavy machine learning libraries.  The chatbot now runs
asynchronously and saves conversation history to a small JSON file.

## Requirements

- Python 3.8+

There are no external dependencies to install.

## Usage

Run the chatbot from the command line:

```bash
python carl_chatbot.py
```

Type your messages after the `You:` prompt. Type `exit`, `quit`, or `bye` to
leave the chat.

### Features

- **Pattern rules** for common greetings and questions
- **Sentiment tracking** that adjusts Carl's mood based on your messages
- **Markov chain fallback** responses trained on a small built‑in corpus
- **Conversation memory** so Carl can reference earlier exchanges
- **Asynchronous interface** for snappier interaction
- **Synonym normalization** for better pattern matching
- **Persistent history** stored in `history.json`

