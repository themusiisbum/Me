import random
import re


def load_rules():
    """Return a list of (pattern, responses) rules."""
    return [
        (re.compile(r"\b(hi|hello|hey)\b", re.I), ["Hey!", "Hello there!", "Hi!"]),
        (re.compile(r"\bhow are you\b", re.I), ["I'm doing great!", "I'm fine, thanks for asking!"]),
        (re.compile(r"\bwhat is your name\b", re.I), ["I'm Carl, nice to meet you."]),
        (re.compile(r"\bwho are you\b", re.I), ["I'm Carl, your friendly chatbot."]),
        (re.compile(r"\bthank you\b", re.I), ["You're welcome!", "No problem!"]),
        # Generic question handler
        (re.compile(r"\?\s*$"), ["That's an interesting question.", "I'm not sure about that."]),
    ]


DEFAULT_RESPONSES = [
    "Tell me more.",
    "Interesting!",
    "I'm not sure I understand, but go on.",
    "Could you elaborate?",
]


def generate_reply(user_input, rules):
    """Generate a reply based on user_input using the provided rules."""
    for pattern, responses in rules:
        match = pattern.search(user_input)
        if match:
            reply = random.choice(responses)
            if match.groups():
                reply = reply.format(*match.groups())
            return reply
    return random.choice(DEFAULT_RESPONSES)


def chat():
    print("Carl: Hello! I'm Carl, your friendly chatbot. Ask me anything!")
    rules = load_rules()
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Carl: Goodbye!")
            break
        reply = generate_reply(user_input, rules)
        print(f"Carl: {reply}")


if __name__ == "__main__":
    chat()
