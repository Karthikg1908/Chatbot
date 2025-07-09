import json
import os
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "train_data.json"
MODEL_FILE = "chatbot_model.pkl"
LOG_FILE = "chat_history.txt"

# Train the chatbot
def train_model():
    if not os.path.exists(DATA_FILE):
        print(f"❌ '{DATA_FILE}' not found. Please create a Q&A dataset.")
        return

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)

    joblib.dump((vectorizer, data), MODEL_FILE)
    print("✅ Training completed and model saved.")

# Save chat history
def log_chat(user_input, bot_response):
    with open(LOG_FILE, "a", encoding="utf-8") as f:  # <-- Fix here
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        f.write(f"{timestamp} You: {user_input}\n")
        f.write(f"{timestamp} Bot: {bot_response}\n\n")


# Start chatbot
def start_chat():
    if not os.path.exists(MODEL_FILE):
        print("❌ Model not trained yet. Run training first.")
        return

    vectorizer, data = joblib.load(MODEL_FILE)
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    X = vectorizer.transform(questions)

    print("\n🤖 Chatbot is ready! Type 'bye' to exit.\n")

    while True:
        user_input = input("You: ").lower()
        if user_input in ["bye", "exit", "quit"]:
            print("Bot: Goodbye! 👋")
            log_chat(user_input, "Goodbye! 👋")
            break

        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X).flatten()

        top_indices = similarity.argsort()[-2:][::-1]
        best_match = top_indices[0]
        confidence = similarity[best_match]

        if confidence < 0.3:
            response = "Sorry, I don't understand. 🤔"
        else:
            response = answers[best_match]

        print("Bot:", response)
        log_chat(user_input, response)

# Main menu
def main():
    print("=== Custom NLP Chatbot ===")
    print("1. Train the chatbot")
    print("2. Chat with the bot")
    choice = input("Choose an option (1 or 2): ")

    if choice == "1":
        train_model()
    elif choice == "2":
        start_chat()
    else:
        print("❌ Invalid choice.")

if __name__ == "__main__":
    main()
