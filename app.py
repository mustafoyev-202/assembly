import streamlit as st
import assemblyai as aai
import os
from dotenv import load_dotenv

# Load the .env file to get the API key
load_dotenv()

# Get the API key from the environment variable
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# Streamlit app title
st.title("Speech to Text with Speaker Diarization and Summary")

# Upload audio file
audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio_file.mp3", "wb") as f:
        f.write(audio_file.getbuffer())

    st.write("Transcribing the audio...")

    # Transcription configuration with speaker diarization enabled
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()

    # Transcribe the uploaded audio with diarization
    transcript = transcriber.transcribe("temp_audio_file.mp3", config=config)

    # Check for transcription errors
    if transcript.status == aai.TranscriptStatus.error:
        st.error(f"Error transcribing audio: {transcript.error}")
    else:
        # Display transcription with speaker labels
        st.write("Transcription completed! Below is the transcribed text with speaker labels:")
        for utterance in transcript.utterances:
            st.write(f"**Speaker {utterance.speaker}:** {utterance.text}")

        # Combine all utterances into one text for further processing
        full_transcript = " ".join([utterance.text for utterance in transcript.utterances])

        # Lemur LLM integration to summarize the transcript
        st.write("Summarizing the transcript with Lemur...")

        prompt = (
            "You are an AI designed to quickly summarize and categorize conversations. "
            "Your task is to provide a brief analysis of the conversation.\n\n"
            "1. Quick Summary: Summarize the main points of the conversation in 2-3 sentences.\n"
            "2. Topic Category: Identify and list the main topics or categories discussed "
            "(e.g., business, personal, technical, etc.).\n"
        )

        try:
            result = transcript.lemur.task(
                prompt, final_model=aai.LemurModel.claude3_5_sonnet
            )

            if result:
                st.write("### Summary of the Transcript:")
                st.write(result.response)
            else:
                st.error("Error generating summary from Lemur.")
        except Exception as e:
            st.error(f"An error occurred while using Lemur: {str(e)}")

        # Optionally, display the full transcript
        if st.checkbox("Show full transcript"):
            st.write(full_transcript)
