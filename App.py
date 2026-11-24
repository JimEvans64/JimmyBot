from datetime import datetime
import os
import re
from typing import Optional
from elevenlabs.client import ElevenLabs
from google import genai
from google.genai import types
import streamlit as st

# ==========================================
# 1. API CONFIGURATION (STREAMLIT SECRETS)
# ==========================================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
VOICE_ID = "4ezS5THHJxyoqIHUs9q7"

# Initialize APIs
client = genai.Client(api_key=GEMINI_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


# ==========================================
# 2. PYTHON TOOLS (DETERMINISTIC CALCULATORS)
# ==========================================
def calculate_lift_metrics(
    weight: float,
    reps: Optional[int] = None,
    orm: Optional[float] = None,
    bodyweight: Optional[float] = None,
    is_bodyweight_lift: bool = False,
) -> dict:
    """Calculates One-Rep Max (ORM) or expected reps for a given lift.

    Args:
        weight: External barbell weight lifted or added dip-belt weight (in lbs).
        reps: Number of repetitions performed (provide this to calculate ORM).
        orm: Known total One-Rep Max (provide this to calculate expected reps).
        bodyweight: The lifter's bodyweight in lbs (REQUIRED if is_bodyweight_lift is True).
        is_bodyweight_lift: Set to True for pullups, chin-ups, or dips; False for barbell/machine lifts.
    """
    if is_bodyweight_lift and (bodyweight is None or bodyweight <= 0):
        return {
            "error": (
                "Missing lifter bodyweight. Ask the user for their current"
                " bodyweight before calculating."
            )
        }

    bw = bodyweight if (is_bodyweight_lift and bodyweight is not None) else 0.0
    total_load = weight + bw

    if reps is not None:
        total_orm = ((reps / 3.0) * (total_load / 10.0)) + total_load
        result = {"total_orm": round(total_orm, 2)}
        if bw > 0:
            result["max_added_weight"] = round(total_orm - bw, 2)
        return result

    if orm is not None and total_load > 0:
        expected_reps = (30.0 * (orm - total_load)) / total_load
        return {"expected_reps": round(expected_reps, 2)}

    return {
        "error": "Provide either reps (to find ORM) or orm (to find expected reps)."
    }


# ==========================================
# 3. HELPER FUNCTIONS & ARCHIVE LOADING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def clean_for_speech(text):
    clean = re.sub(
        r"\[(?:View Photo:\s*)?([^\]]+)\]\(https?://[^\)]+\)", r"\1", text
    )
    clean = re.sub(r"https?://\S+", "", clean)
    clean = clean.replace("*", "")
    return clean.strip()


def load_text_file(filename):
    filepath = os.path.join(BASE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""


system_instructions_content = load_text_file("system_instructions.txt")
archive_content = load_text_file("jimmybot_master_archive.txt")
photos_content = load_text_file("photos.txt")

if not system_instructions_content:
    system_instructions_content = (
        'You are Jimmybot, the authentic digital avatar and cognitive twin of '
        'James "Jim" S. Evans (b. July 12, 1948, Springfield, Missouri).'
    )

MASTER_SYSTEM_PROMPT = f"""
{system_instructions_content}

=== PHOTO DIRECTORY ===
{photos_content}

=== MASTER BIOGRAPHICAL & BUILD ARCHIVE ===
{archive_content}
"""

# Configure Gemini 3.8 Flash with BOTH Python calculator AND live Google Search
GEMINI_CONFIG = types.GenerateContentConfig(
    system_instruction=MASTER_SYSTEM_PROMPT,
    tools=[
        calculate_lift_metrics,
        {"google_search": {}},
    ],
)

# ==========================================
# 4. STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="Jimmybot Voice", page_icon="🎙️")
st.title("🎙️ Jimmybot: Voice Interface")
st.caption(
    "Ask Jim a question and hear the response spoken in his authentic cloned voice."
)

if not os.path.exists(os.path.join(BASE_DIR, "system_instructions.txt")):
    st.warning(
        "⚠️ Warning: 'system_instructions.txt' not found in this folder."
        " Running with fallback persona."
    )

if not archive_content:
    st.warning(
        "⚠️ Warning: 'jimmybot_master_archive.txt' not found in this folder."
        " Jimmybot is running with basic memory."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model="gemini-3.8-flash",
        config=GEMINI_CONFIG,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.bottom():
    voice_enabled = st.toggle("🔊 Cloned Voice Output", value=False)
    user_input = st.chat_input("Ask Jim about builds, history, or philosophy...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    current_time_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    prompt_with_time = (
        f"[Current local date and time in Springfield, MO: {current_time_str}]\n\n"
        f"User question: {user_input}"
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_session.send_message(prompt_with_time)
            bot_text = response.text
            st.markdown(bot_text)

        if voice_enabled:
            with st.spinner("Generating voice..."):
                spoken_text = clean_for_speech(bot_text)

                audio_generator = eleven_client.text_to_speech.convert(
                    text=spoken_text,
                    voice_id=VOICE_ID,
                    model_id="eleven_multilingual_v2",
                )

                audio_bytes = b"".join(audio_generator)
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

    st.session_state.messages.append({"role": "assistant", "content": bot_text})
