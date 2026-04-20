from src.paths import CHATBOT_LOG_PATH
from src.chatbot import run_safe_chatbot_cli

def main():
    with open(CHATBOT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("# SAFE Chatbot Conversation Log\n\n")
    run_safe_chatbot_cli()

if __name__ == "__main__":
    main()