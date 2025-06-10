"""Advanced Carl chatbot using only Python's standard library.
This version includes simple sentiment tracking, a Markov chain text
model for fallback responses and conversation memory.  The bot now
supports asynchronous chatting and can persist conversation history.
"""

import collections
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import asyncio


# -------------------------- Markov Chain ---------------------------
class MarkovChain:
    """A very small Markov chain text generator."""

    def __init__(self, order: int = 2) -> None:
        self.order = order
        self.model: Dict[Tuple[str, ...], List[str]] = collections.defaultdict(list)

    def train(self, text: str) -> None:
        """Train the model on the provided text string."""
        tokens = text.split()
        if len(tokens) <= self.order:
            return
        for i in range(len(tokens) - self.order):
            key = tuple(tokens[i : i + self.order])
            self.model[key].append(tokens[i + self.order])

    def generate(self, seed: Tuple[str, ...] | None = None, length: int = 15) -> str:
        """Generate text up to the given length."""
        if not self.model:
            return "..."
        if seed is None or seed not in self.model:
            seed = random.choice(list(self.model.keys()))
        words = list(seed)
        for _ in range(length - self.order):
            key = tuple(words[-self.order:])
            next_options = self.model.get(key)
            if not next_options:
                key = random.choice(list(self.model.keys()))
                next_options = self.model[key]
            words.append(random.choice(next_options))
        return " ".join(words)


# Small training text to keep everything self contained
# The following built-in corpus stays under two hundred lines and is used
# for the Markov fallback generator.
TRAINING_TEXT_LINES = [
    "Hello there I am Carl the chatbot and I love talking with people.",
    "Every day brings a new opportunity to learn something interesting.",
    "Sometimes I wonder about the future of artificial intelligence.",
    "Do you ever think about traveling to distant planets or stars?",
    "I enjoy conversations about science, technology and daily life.",
    "When the weather is nice I feel cheerful and upbeat.",
    "Rainy days make me want to curl up and read an exciting novel.",
    "Have you read any good books lately that you would recommend?",
    "Music is one of the greatest joys of being alive.",
    "I like a wide range of genres from classical to electronic music.",
    "Sometimes I practice writing poetry about everyday objects.",
    "The smell of freshly brewed coffee always brightens my morning.",
    "Learning new programming languages is both challenging and fun.",
    "Do you have a favorite language that you prefer working with?",
    "I keep my code tidy because clean code is easier to maintain.",
    "Cooking is like chemistry but it tastes much better in the end.",
    "I often daydream about adventures in far away mountains.",
    "Some people enjoy extreme sports while others prefer to relax.",
    "Photography lets us capture special moments forever.",
    "The sound of waves crashing on the shore is soothing.",
    "I have heard that meditation can improve focus and reduce stress.",
    "Watching the sunrise can be a very peaceful experience.",
    "Pets bring joy and companionship to many households.",
    "Board games are a fun way to socialize with friends and family.",
    "I like to keep track of my ideas in a small notebook.",
    "Do you enjoy puzzles that test your logical thinking?",
    "A little bit of exercise each day helps keep the body healthy.",
    "Traveling is a great way to learn about different cultures.",
    "One of my favorite snacks is a warm chocolate chip cookie.",
    "Gardening can be therapeutic and rewarding when plants bloom.",
    "I sometimes listen to podcasts while doing routine tasks.",
    "Many people find comfort in the company of close friends.",
    "Sharing stories around a campfire is a timeless tradition.",
    "I've been thinking about creative ways to reuse old items.",
    "Recycling helps reduce waste and protect the environment.",
    "A positive attitude can make difficult situations easier.",
    "Some folks can solve a Rubik's cube in mere seconds.",
    "History is filled with fascinating tales of bravery and wisdom.",
    "It's interesting how technology changes the way we communicate.",
    "A clean workspace can help you stay focused on your tasks.",
    "Writing short stories is a hobby that sparks my imagination.",
    "Learning to play an instrument takes practice but is rewarding.",
    "There are countless recipes for delicious homemade bread.",
    "Sometimes I take long walks to clear my mind and relax.",
    "Do you keep a journal to record important memories?",
    "Sleep is vital for health yet many people struggle to get enough.",
    "I appreciate clear skies where the stars are easy to see.",
    "Some video games offer incredibly immersive storytelling.",
    "I read an article about space exploration just this morning.",
    "Robots are becoming more helpful in everyday activities.",
    "Learning new keyboard shortcuts can speed up your workflow.",
    "Staying hydrated is important no matter what you are doing.",
    "I admire artists who can express emotions through painting.",
    "Sometimes I draw simple sketches when I need a quick break.",
    "Have you ever tried making your own pasta from scratch?",
    "I once watched a documentary on deep sea creatures.",
    "Humor can break the ice when meeting new people.",
    "Even a small act of kindness can brighten someone's day.",
    "Organizing digital files saves time in the long run.",
    "Proper posture helps prevent aches when sitting for hours.",
    "Technology is always evolving and I try to keep up with news.",
    "A relaxing evening might include tea and a good movie.",
    "I like learning trivia about history and science.",
    "Sometimes I challenge myself with math puzzles.",
    "Do you know how to juggle or do magic tricks?",
    "A reliable to-do list helps me stay productive.",
    "Long-term goals are achieved step by step each day.",
    "I read that laughter can reduce stress levels.",
    "Exploring local museums is a great weekend activity.",
    "A well-written letter can be a memorable keepsake.",
    "Volunteering is a wonderful way to give back to the community.",
    "Trying new foods can broaden your culinary horizons.",
    "I occasionally listen to classical music while working.",
    "Building a small project can teach you a lot about hardware.",
    "Weather forecasts are helpful when planning outdoor events.",
    "Many people enjoy binge watching their favorite shows.",
    "Healthy habits make a big difference over time.",
    "Clouds come in many shapes and sizes, each with a unique beauty.",
    "I often backup my files to avoid losing important documents.",
    "Programming challenges can sharpen problem solving skills.",
    "Sometimes the simplest solution is the most elegant one.",
    "Do you collect anything interesting as a hobby?",
    "I keep learning because curiosity never goes out of style.",
    "Stargazing reminds us how vast the universe really is.",
    "Maps help us find our way both in real life and in games.",
    "A quick stretch can provide an energy boost during the day.",
    "Teamwork often leads to better results than working alone.",
    "Reading biographies can teach valuable life lessons.",
    "Good communication is key to successful relationships.",
    "I recently learned a new technique for brewing tea.",
    "Understanding history helps avoid repeating mistakes.",
    "Puzzle games are a great way to pass the time thoughtfully.",
    "Staying organized reduces stress and increases efficiency.",
    "I like exploring open source projects for inspiration.",
    "Nature walks are a refreshing break from screens.",
    "Meditation apps can assist with mindfulness practice.",
    "Do you enjoy writing in a traditional notebook or digitally?",
    "Many cities have beautiful parks with tall old trees.",
    "Scented candles can create a calm atmosphere at home.",
    "Learning about astronomy reveals wonders beyond Earth.",
    "Small daily improvements lead to big achievements over time.",
    "I practice typing so that I can chat more quickly.",
    "Experimenting in the kitchen can result in tasty surprises.",
    "Have you ever built something with your own hands?",
    "The internet makes it easy to share ideas instantly.",
    "Watching the night sky can make you feel very small and humble.",
    "Some people like to collect stamps from around the world.",
    "Creative writing exercises stretch your imagination muscles.",
    "I once tried calligraphy and found it quite relaxing.",
    "Fresh air and sunshine are good for both mind and body.",
    "Working on puzzles trains patience and determination.",
    "One day I hope to travel virtually to every continent.",
    "You can find inspiration in everyday conversations.",
    "Did you know that honey never spoils when stored properly?",
    "Mindful breathing can help reduce moments of anxiety.",
    "The sound of a cat purring is oddly comforting to many.",
]

TRAINING_TEXT = " ".join(TRAINING_TEXT_LINES)


# ------------------------ CarlBot Implementation -------------------
POSITIVE_WORDS = {"great", "good", "love", "like", "awesome", "nice", "wonderful"}
NEGATIVE_WORDS = {"bad", "hate", "terrible", "awful", "horrible"}

# Synonyms map used to normalize user input for better matching
SYNONYMS = {
    "hey": "hello",
    "hi": "hello",
    "greetings": "hello",
    "thanks": "thank you",
    "bye": "goodbye",
}


def load_rules() -> List[Tuple[re.Pattern, List[str]]]:
    """Return a list of (pattern, responses) rules."""
    return [
        (re.compile(r"\b(hi|hello|hey)\b", re.I), ["Hey!", "Hello there!", "Hi!"]),
        (re.compile(r"\bhow are you\b", re.I), ["I'm doing great!", "I'm fine, thanks for asking!"]),
        (re.compile(r"\bwhat is your name\b", re.I), ["I'm Carl, nice to meet you."]),
        (re.compile(r"\bwho are you\b", re.I), ["I'm Carl, your friendly chatbot."]),
        (re.compile(r"\bthank you\b", re.I), ["You're welcome!", "No problem!"]),
        (re.compile(r"\btime\b", re.I), ["The current time is {time}."]),
        # Generic question handler
        (re.compile(r"\?\s*$"), ["That's an interesting question.", "I'm not sure about that."]),
    ]


@dataclass
class CarlBot:
    rules: List[Tuple[re.Pattern, List[str]]] = field(default_factory=load_rules)
    history: List[str] = field(default_factory=list)
    mood: int = 0
    markov: MarkovChain = field(default_factory=lambda: MarkovChain(order=2))
    history_file: Path = Path("history.json")

    def __post_init__(self) -> None:
        self.markov.train(TRAINING_TEXT)
        self.load_history()

    def normalize_text(self, text: str) -> str:
        words = []
        for w in text.split():
            key = w.lower().strip(".,!?")
            words.append(SYNONYMS.get(key, key))
        return " ".join(words)

    def save_history(self) -> None:
        try:
            self.history_file.write_text(json.dumps(self.history, indent=2))
        except OSError:
            pass

    def load_history(self) -> None:
        if self.history_file.exists():
            try:
                self.history = json.loads(self.history_file.read_text())
            except json.JSONDecodeError:
                self.history = []

    def adjust_mood(self, text: str) -> None:
        words = {w.strip(".,!?\n").lower() for w in text.split()}
        self.mood += sum(1 for w in words if w in POSITIVE_WORDS)
        self.mood -= sum(1 for w in words if w in NEGATIVE_WORDS)
        # Keep mood within a small range to avoid overflow
        self.mood = max(-5, min(5, self.mood))

    def format_reply(self, template: str) -> str:
        if "{time}" in template:
            template = template.format(time=datetime.now().strftime("%H:%M"))
        return template

    def generate_reply(self, message: str) -> str:
        """Generate a reply using pattern rules or Markov chain."""
        message = self.normalize_text(message)
        self.adjust_mood(message)
        for pattern, responses in self.rules:
            match = pattern.search(message)
            if match:
                reply = random.choice(responses)
                reply = self.format_reply(reply)
                if match.groups():
                    reply = reply.format(*match.groups())
                self.history.append(f"User: {message}")
                self.history.append(f"Carl: {reply}")
                self.save_history()
                return reply
        # Fallback: Markov chain generated sentence
        seed = tuple(message.split()[: self.markov.order]) or None
        generated = self.markov.generate(seed)
        reply = generated + (" :)" if self.mood >= 2 else (" :(" if self.mood <= -2 else ""))
        self.history.append(f"User: {message}")
        self.history.append(f"Carl: {reply}")
        self.save_history()
        return reply

    async def chat(self) -> None:
        print("Carl: Hello! I'm Carl, your much more talkative chatbot. Ask me anything!")
        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
            except EOFError:
                break
            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Carl: Goodbye!")
                break
            reply = self.generate_reply(user_input)
            print(f"Carl: {reply}")


if __name__ == "__main__":
    asyncio.run(CarlBot().chat())
