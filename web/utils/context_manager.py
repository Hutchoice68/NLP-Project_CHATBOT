class ContextMemory:
    def __init__(self, max_turns=3):
        self.history = []
        self.max_turns = max_turns

    def add(self, user_input, intent):
        self.history.append({"text": user_input, "intent": intent})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context_text(self):
        return " ".join([h["text"] for h in self.history])