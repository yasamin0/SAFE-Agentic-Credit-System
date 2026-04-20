# src/chat_cli.py

# Path where chatbot conversations are stored as a Markdown log
from src.paths import CHATBOT_LOG_PATH

# Imports the standalone interactive SAFE chatbot interface
from src.chatbot import run_safe_chatbot_cli


def main():
    # Reset the chatbot log file at the beginning of each standalone chat session
    # so the new conversation starts with a clean log
    with open(CHATBOT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("# SAFE Chatbot Conversation Log\n\n")

    # Launch the interactive SAFE chatbot loop
    run_safe_chatbot_cli()


if __name__ == "__main__":
    # Entry point for running the chatbot as a standalone module:
    # python -m src.chat_cli
    main()