import streamlit as st
import whisper
import tempfile
import os

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Streamlit UI
st.title("🎤 Audio to Text Transcription (Whisper - Local Model)")

uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_audio.read())
        temp_audio_path = tmp.name

    st.audio(uploaded_audio, format='audio/mp3')

    with st.spinner("Transcribing... Please wait"):
        result = model.transcribe(temp_audio_path)
        st.success("✅ Transcription completed!")

    st.subheader("📝 Transcribed Text:")
    st.write(result["text"])

    os.remove(temp_audio_path)
