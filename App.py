import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration (Browser Tab)
st.set_page_config(page_title="JimmyBot", page_icon="🛠️", layout="centered")

# 2. On-Screen Header
st.title("JimmyBot")
st.caption("Jim Evans' Digital Twin")

# 3. Secure API Initialization
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("API Key not found. Please set GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# 4. Load System Instructions, Photo Directory, and Master Archive
@st.cache_data
def load_context():
    with open("system_instructions.txt", "r", encoding="utf-8") as f:
        instructions = f.read()
    with open("photos.txt", "r", encoding="utf-8") as f:
        photos = f.read()
    with open("jimmybot_master_archive.txt", "r", encoding="utf-8") as f:
        archive = f.read()
    
    full_system_prompt = (
        f"{instructions}\n\n"
        f"=== PHOTO DIRECTORY ===\n{photos}\n\n"
        f"=== GROUND-TRUTH MASTER ARCHIVE ===\n{archive}"
    )
    return full_system_prompt

system_instruction = load_context()
# 5. Initialize Chat Session State & Sidebar Controls
initial_greeting = "Good day, my friend. Let's reminisce or solve the world's problems, or at least tackle our own. Shoot."

# Sidebar for controls
with st.sidebar:
    st.markdown("### Controls")
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Ensure session state always starts with the initial greeting
if not st.session_state.get("messages"):
    st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]

# 6. Display Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Chat Input & Response Loop
if user_input := st.chat_input("Ask a question, troubleshoot a build, or explore a memory..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Format history for the Gemini API (Must start with a 'user' turn)
    api_contents = []
    for msg in st.session_state.messages:
        if not api_contents and msg["role"] == "assistant":
            continue  # Skip initial UI greeting from the API payload
        role = "model" if msg["role"] == "assistant" else "user"
        api_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

    # Generate Streamed Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        response_stream = client.models.generate_content_stream(
            model="gemini-3.8-flash",
            contents=api_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.5, # Balanced for wit, warmth, and precision
                tools=[{"google_search": {}}], # Live search grounding for current events
            )
        )

        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

    # Save assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
