import gradio as gr
import requests


# --- Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# --- Conversation State ---
conversation_history = []
current_question_index = 0
questions = [
    "What is your Name?",
    "What is your Email?",
    "What is your Phone number?",
    "How many years of experience do you have?",
    "Where are you located?",
    "What is your desired role?",
    "What is your tech stack?"
]

# --- Simulated Data Storage ---
candidate_data = {}

# --- Prompt Templates ---
def build_prompt(user_input, history):
    # You can improve this prompt over time
    context = "\n".join([f"User: {msg['user']}\nAI: {msg['bot']}" for msg in history])
    return f"""
You are TalentScout, a friendly and professional hiring assistant chatbot.

Your tasks:
- Greet the candidate.
- Collect their details (Name, Email, Phone, Experience, Location, Desired Role, Tech Stack).
- Ask 3-5 technical questions based on their tech stack.
- Handle unexpected inputs gracefully.
- Say goodbye politely if user wants to exit.

Conversation so far:
{context}

User: {user_input}
TalentScout:"""

# --- LLaMA 3 via Ollama ---
def query_llama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

# --- Chatbot Logic ---
def chat(user_input):
    global current_question_index

    if user_input.lower() in ["exit", "quit", "bye"]:
        bot_reply = "Thank you for chatting with TalentScout. We’ll get back to you soon. Goodbye!"
        conversation_history.append({"user": user_input, "bot": bot_reply})
        return [(user_input, bot_reply)]

    if current_question_index < len(questions):
        question = questions[current_question_index]
        bot_reply = question
        candidate_data[question] = user_input  # Store response
        current_question_index += 1
    else:
        # Once all questions are asked, proceed with the main chatbot logic
        prompt = build_prompt(user_input, conversation_history)
        bot_reply = query_llama(prompt)

    conversation_history.append({"user": user_input, "bot": bot_reply})
    return [(turn["user"], turn["bot"]) for turn in conversation_history]

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 TalentScout - Hiring Assistant Chatbot")
    gr.Markdown(
        "### Disclaimer: Your data will only be used for this session and will not be stored permanently."
    )

    chatbot = gr.Chatbot(value=[
        ("TalentScout", "Hello! I am TalentScout, your friendly hiring assistant chatbot. "
                        "I will guide you through a few questions to collect your details and discuss your skills. "
                        "Feel free to type 'exit' anytime to end the chat.")
    ])
    msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")

    def respond(user_message):
        return chat(user_message)

    msg.submit(respond, inputs=msg, outputs=chatbot)

# --- Run ---
if __name__ == "__main__":
    demo.launch()
    print(candidate_data)

# This code is a simple chatbot using Gradio and LLaMA 3 via Ollama.