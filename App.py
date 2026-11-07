import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(page_title="Jimmybot", page_icon="🛠️", layout="centered")
st.title("Jimmybot: Cognitive Twin of Jim Evans")
st.caption("Mathematics & CS Educator | Welder | Passive-Solar Builder | 50,000-Mile Runner")

# 2. Secure API Initialization
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API Key not found. Please set GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# 3. Load System Instructions and Master Archive
@st.cache_data
def load_context():
    with open("system_instructions.txt", "r", encoding="utf-8") as f:
        instructions = f.read()
    with open("Jimmybot_Master_Archive.txt", "r", encoding="utf-8") as f:
        archive = f.read()
    
    # Combine the core instructions with the master archive
    full_system_prompt = f"{instructions}\n\n=== GROUND-TRUTH MASTER ARCHIVE ===\n{archive}"
    return full_system_prompt

system_instruction = load_context()

# 4. Initialize Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Kick off with a grounded opening greeting
    initial_greeting = "The full workbench is laid out, from passive solar framing down to forty-five years of running logs. What are we getting into today?"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

# 5. Display Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Chat Input & Response Loop
if user_input := st.chat_input("Ask a question, troubleshoot a build, or explore a memory..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Format history for the Gemini API
    # Mapping 'assistant' to 'model' for the API
    api_contents = []
    for msg in st.session_state.messages:
        role = "model" if msg["role"] == "assistant" else "user"
        api_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    # Generate Streamed Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        response_stream = client.models.generate_content_stream(
            model="gemini-2.5-pro", # or your chosen model
            contents=api_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4, # Moderate temperature keeps reasoning consistent and grounded
            )
        )

        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

    # Save assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})