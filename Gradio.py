import gradio as gr
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import random

# Load fine-tuned model and tokenizer
model_path = "/content/drive/MyDrive/Colab Notebooks/Medical_chatbot/healthcare-chatbot-model"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Medical keywords for filtering
medical_keywords = [
    "symptom", "diagnose", "treatment", "medicine", "disease", "doctor",
    "covid", "cancer", "diabetes", "bipolar", "stroke", "fever", "infection",
    "pain", "mental", "health", "hospital", "vaccine", "prescription"
]

# Greeting keywords
greeting_keywords = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "greetings", "what's up", "whats up", "how are you", "sup"
]

# Greeting responses
greeting_responses = [
    "Hello! 👋 I'm your medical assistant. How can I help you with your health questions today?",
    "Hi there! 🩺 I'm here to help with any medical or health-related questions you might have.",
    "Greetings! 😊 I'm your healthcare chatbot. Feel free to ask me about symptoms, treatments, or general health information.",
    "Hello! 🏥 Nice to meet you! I'm ready to assist with your medical inquiries.",
    "Hi! 👨‍⚕️ I'm your AI medical assistant. What health topic would you like to discuss today?"
]

fun_facts = [
    "Did you know? The human brain has around 86 billion neurons!",
    "Fun fact: Laughing is good for your heart and can reduce stress.",
    "Tip: Drinking water can improve cognitive performance."
]

def is_greeting(q):
    return any(keyword in q.lower() for keyword in greeting_keywords)

def is_medical_question(q):
    return any(keyword in q.lower() for keyword in medical_keywords)

def generate_answer(question):
    question_lower = question.lower().strip()

    # Handle greetings
    if is_greeting(question_lower):
        return random.choice(greeting_responses)

    # Handle medical questions
    if is_medical_question(question_lower):
        input_text = "healthcare question: " + question
        input_ids = tokenizer(input_text, return_tensors="tf", padding=True, truncation=True).input_ids

        output = model.generate(
            input_ids,
            max_length=128,
            num_beams=4,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            early_stopping=True
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response[0].upper() + response[1:]  # Capitalize first letter

    # Handle non-medical questions
    return random.choice(fun_facts) + "\n\n🩺 Please ask a healthcare-related question or feel free to greet me!"

# Enhanced chatbot function with immediate message display
def chatbot_respond(message, history):
    if message.strip() == "":
        return history, ""

    # Add user message immediately to chat
    new_history = history + [(message, "Thinking...")]

    # Generate response
    response = generate_answer(message)

    # Update the last entry with actual response
    new_history[-1] = (message, response)

    return new_history, ""

# Function for example questions with immediate display
def handle_example_question(question, history):
    # Add question immediately to chat
    new_history = history + [(question, "Thinking...")]

    # Generate response
    response = generate_answer(question)

    # Update with actual response
    new_history[-1] = (question, response)

    return new_history

def clear_chat():
    return []

# Enhanced CSS with medical color scheme
medical_css = """
/* Medical color scheme with professional appearance */
body {
    background: linear-gradient(135deg, #e8f5e8 0%, #f0f8ff 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    padding: 20px !important;
}

.gradio-container {
    background: linear-gradient(135deg, #e8f5e8 0%, #f0f8ff 100%) !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
}

/* Header styling */
.markdown h1 {
    color: #2c5530 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
}

/* Chat container styling */
.chatbot {
    background: #ffffff !important;
    border: 2px solid #4a90a4 !important;
    border-radius: 15px !important;
    box-shadow: 0 4px 15px rgba(74, 144, 164, 0.2) !important;
}

/* Chat messages styling - WhatsApp-like */
.message.user {
    background: #dcf8c6 !important;
    border-radius: 18px 18px 4px 18px !important;
    margin: 5px 0 !important;
    padding: 8px 12px !important;
    max-width: 80% !important;
    margin-left: auto !important;
    border: 1px solid #b7e5a1 !important;
    word-wrap: break-word !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
}

.message.bot {
    background: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 18px 18px 18px 4px !important;
    margin: 5px 0 !important;
    padding: 8px 12px !important;
    max-width: 80% !important;
    margin-right: auto !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    word-wrap: break-word !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
}

/* Fix for chat text display */
.chatbot .message, .chatbot .message p {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
    word-break: normal !important;
    hyphens: none !important;
    writing-mode: horizontal-tb !important;
    text-orientation: mixed !important;
}

/* Ensure proper text flow in chat bubbles */
.chatbot .wrap {
    white-space: normal !important;
    word-wrap: break-word !important;
}

.chatbot .message-wrap {
    white-space: normal !important;
    word-wrap: break-word !important;
    display: block !important;
}

/* Input field styling */
.textbox input {
    background: #ffffff !important;
    border: 2px solid #4a90a4 !important;
    border-radius: 25px !important;
    padding: 12px 18px !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}

.textbox input:focus {
    border-color: #2c5530 !important;
    box-shadow: 0 0 10px rgba(44, 85, 48, 0.3) !important;
}

/* Button styling */
.btn-primary {
    background: linear-gradient(135deg, #4a90a4 0%, #2c5530 100%) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 24px !important;
    color: white !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 10px rgba(74, 144, 164, 0.3) !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(74, 144, 164, 0.4) !important;
}

.btn-secondary {
    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 24px !important;
    color: white !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.btn-secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(220, 53, 69, 0.4) !important;
}

/* Sidebar buttons */
.sidebar-btn {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
    border: 2px solid #4a90a4 !important;
    border-radius: 15px !important;
    padding: 10px 15px !important;
    margin: 5px 0 !important;
    color: #2c5530 !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.sidebar-btn:hover {
    background: linear-gradient(135deg, #4a90a4 0%, #2c5530 100%) !important;
    color: white !important;
    transform: translateX(5px) !important;
}

/* Example question buttons */
.example-btn {
    background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%) !important;
    border: 2px solid #4a90a4 !important;
    border-radius: 20px !important;
    padding: 10px 15px !important;
    margin: 5px !important;
    color: #2c5530 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

.example-btn:hover {
    background: linear-gradient(135deg, #4a90a4 0%, #2c5530 100%) !important;
    color: white !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 10px rgba(74, 144, 164, 0.3) !important;
}

/* About section styling */
.about-section {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid #4a90a4 !important;
    border-radius: 15px !important;
    padding: 15px !important;
    margin: 10px 0 !important;
    box-shadow: 0 2px 8px rgba(74, 144, 164, 0.2) !important;
}

/* Loading animation for "Thinking..." */
@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

.thinking {
    animation: pulse 1.5s infinite;
    color: #4a90a4 !important;
    font-style: italic !important;
}
"""

# Create the interface
with gr.Blocks(css=medical_css, theme=gr.themes.Soft(primary_hue="teal", secondary_hue="green")) as demo:

    # Header
    gr.Markdown("""
    <div style="text-align: center; padding: 15px;">
        <h1 style="color: #2c5530; font-size: 2em; margin-bottom: 8px;">🏥 Medical Knowledge Chatbot</h1>
        <p style="color: #4a90a4; font-size: 16px; font-weight: 500;">
            Your AI assistant for understanding <b>health</b> and <b>medicine</b>
        </p>
        <div style="width: 80px; height: 2px; background: linear-gradient(90deg, #4a90a4, #2c5530); margin: 15px auto; border-radius: 2px;"></div>
    </div>
    """)

    with gr.Row():
        # Main chat area
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(
                label="💬 Medical Assistant Chat",
                height=400,
                show_copy_button=True,
                bubble_full_width=False,
                avatar_images=["👤", "🩺"]
            )

            msg = gr.Textbox(
                placeholder="Type your medical question here and press Enter...",
                label="💬 Ask Your Question",
                lines=1,
                max_lines=2,
                container=True
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 Send Message", variant="primary", scale=2)
                clear_btn = gr.Button("🧹 Clear Chat", variant="secondary", scale=1)

        # Sidebar
        with gr.Column(scale=1):
            gr.Markdown("""
            <div class="about-section">
                <h3 style="color: #2c5530; margin-bottom: 15px;">📚 Medical Topics</h3>
            </div>
            """)

            topic_btns = []
            topics = [
                ("🩺 Symptoms", "symptoms"),
                ("🦠 Diseases", "diseases"),
                ("💊 Treatments", "treatments"),
                ("💉 Medications", "medications"),
                ("🧠 Mental Health", "mental health")
            ]

            for topic_name, topic_key in topics:
                btn = gr.Button(topic_name, elem_classes="sidebar-btn")
                topic_btns.append(btn)

            gr.Markdown("""
            <div class="about-section">
                <h3 style="color: #2c5530; margin-bottom: 10px;">ℹ️ About</h3>
                <p style="color: #555; font-size: 14px; line-height: 1.5;">
                    This chatbot uses a fine-tuned <b>T5 model</b> to provide medical information.<br><br>
                    <strong>⚠️ Important:</strong> This is for educational purposes only and should not replace professional medical advice.
                </p>
            </div>
            """)

    # Example questions section
    gr.Markdown("""
    <div style="text-align: center; margin: 20px 0 15px 0;">
        <h3 style="color: #2c5530; font-size: 1.3em;">💡 Try These Example Questions</h3>
        <p style="color: #666; margin-bottom: 15px; font-size: 14px;">Click on any question to get started</p>
    </div>
    """)

    example_questions = [
        "What are the symptoms of diabetes?",
        "How is hypertension treated?",
        "What is bipolar disorder?",
        "What medicine is used for asthma?",
        "Is COVID-19 contagious?"
    ]

    with gr.Row():
        example_btns = []
        for question in example_questions:
            btn = gr.Button(question, elem_classes="example-btn", size="sm")
            example_btns.append(btn)

    # Event handlers
    def submit_message(message, history):
        return chatbot_respond(message, history)

    # Handle regular message submission
    submit_btn.click(
        fn=submit_message,
        inputs=[msg, chatbot_ui],
        outputs=[chatbot_ui, msg]
    )

    # Handle Enter key press
    msg.submit(
        fn=submit_message,
        inputs=[msg, chatbot_ui],
        outputs=[chatbot_ui, msg]
    )

    # Handle example question clicks
    for i, btn in enumerate(example_btns):
        btn.click(
            fn=handle_example_question,
            inputs=[gr.State(example_questions[i]), chatbot_ui],
            outputs=[chatbot_ui]
        )

    # Handle clear button
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot_ui]
    )

    # Footer
    gr.Markdown("""
    <div style="text-align: center; margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.7); border-radius: 15px;">
        <p style="color: #666; font-size: 13px;">
            🏥 <strong>Medical Knowledge Chatbot</strong> | Powered by AI for Educational Purposes
        </p>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True, share=True)
