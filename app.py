from openai import OpenAI
from autogen import ConversableAgent, GroupChat, GroupChatManager
import gradio as gr
import os
import warnings
import logging
from dotenv import load_dotenv
 
# ---- Suppress warnings ----
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)
 
# ---- Load environment variables ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_ID = os.getenv("MODEL_ID", "gpt-4o-mini")
 
# ---- Initialize LLM config ----
llm_config = {"config_list": [{"model": MODEL_ID, "api_key": OPENAI_API_KEY}]}
 
# ---- Define Agents ----
patient_agent = ConversableAgent(
    name="patient",
    system_message="You describe your emotions and mental health concerns.",
    llm_config=llm_config
)
 
emotion_analysis_agent = ConversableAgent(
    name="emotion_analysis",
    system_message=(
        "You analyze the user's emotions based on their input. "
        "Do not provide treatment or self-care advice. "
        "Instead, just summarize the dominant emotions they may be experiencing."
    ),
    llm_config=llm_config
)
 
therapy_recommendation_agent = ConversableAgent(
    name="therapy_recommendation",
    system_message=(
        "You suggest relaxation techniques and self-care methods "
        "only based on the analysis from the Emotion Analysis Agent. "
        "Do not analyze emotions—just give recommendations based on the prior response."
    ),
    llm_config=llm_config
)
 
# ---- Create Group Chat ----
groupchat = GroupChat(
    agents=[emotion_analysis_agent, therapy_recommendation_agent],
    messages=[],
    max_round=3,
    speaker_selection_method="round_robin"
)
manager = GroupChatManager(name="manager", groupchat=groupchat)
 
# ---- Core logic function ----
 
 
def chat_response(user_input):
    """Handles the full AI interaction flow and captures the final response."""
    try:
        # Clear old messages
        groupchat.messages.clear()
 
        # Run conversation
        patient_agent.initiate_chat(
            manager,
            message=f"I have been feeling {user_input}. Can you help?"
        )
 
        # Capture conversation history (may be objects, not dicts)
        messages = getattr(groupchat, "messages", [])
 
        if not messages:
            return "No messages captured from group chat."
 
        responses = []
        for m in messages:
            # --- Different autogen versions have different message formats ---
            # 1️⃣ Case: dict-based
            if isinstance(m, dict):
                name = m.get("name", "Unknown")
                content = m.get("content", "")
                if content and name not in ["patient"]:
                    responses.append(f"**{name}**: {content}")
 
            # 2️⃣ Case: object-based (Message class)
            elif hasattr(m, "content") and hasattr(m, "name"):
                if m.name != "patient":
                    responses.append(f"**{m.name}**: {m.content}")
 
        if responses:
            return "\n---\n".join(responses)
        else:
            # fallback — use the last message if nothing else matched
            last_msg = messages[-1] if messages else None
            if last_msg and hasattr(last_msg, "content"):
                return str(last_msg.content)
            return "No AI response found."
 
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
 
 
# ---- Gradio UI ----
 
 
def gradio_interface(user_input):
    return chat_response(user_input)
 
 
with gr.Blocks(theme=gr.themes.Soft(), title="AI Mental Health Assistant") as demo:
    gr.Markdown(
        "## 🧠 AI Mental Health Chatbot\nDescribe how you're feeling, and the AI will analyze and recommend self-care tips.")
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="How are you feeling today?",
                placeholder="e.g., I’ve been feeling anxious and tired lately..."
            )
            submit_btn = gr.Button("Analyze & Recommend")
        with gr.Column():
            output = gr.Textbox(label="AI Response", lines=10)
 
    submit_btn.click(fn=gradio_interface, inputs=user_input, outputs=output)
 
# ---- Launch the Gradio app ----
if __name__ == "__main__":
    demo.launch()