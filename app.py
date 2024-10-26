import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from openai import OpenAI
from utils.functions import (
    get_vector_store,
    get_response,
    text_to_audio,
    autoplay_audio,
    speech_to_text,
)
from my_component import chat_input_widget

# Load environment variables
load_dotenv()
client = OpenAI()

# Application layout
def main():
    st.set_page_config("Vet AI Assistant", page_icon="💼")

    # Load CSS
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    user_query = None

    # Display the custom widget and get the response
    response = chat_input_widget()

    if response:
        if 'text' in response:
            user_query = response['text']
        elif 'audioFile' in response:  # Audio file received as binary data
            with st.spinner("Transcribing audio..."):
                audio_file_bytes = response['audioFile']  # Extract byte array

                # Write byte data to a .wav file
                temp_audio_path = "temp_audio.wav"
                with open(temp_audio_path, "wb") as f:
                    f.write(bytes(audio_file_bytes))  # Convert list of bytes to binary

                # Transcribe the audio file
                transcript = speech_to_text(temp_audio_path)

                # Clean up the temporary file
                os.remove(temp_audio_path)

                user_query = transcript

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(
                content="""
                    Welcome to the Veterinary AI Assistant Chat! I am here to help with veterinary medicine questions,
                    including diagnoses, symptoms, food formulations, surgeries, and more.
                    How can I assist you today? 🥰
                """
            )
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store()

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="👩‍⚕️"):
                st.write(message.content)

    # Process user input if available
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human", avatar="👩‍⚕️"):
            st.markdown(user_query)

        # Get AI response
        with st.chat_message("AI", avatar="🤖"):
            response = st.write_stream(get_response(user_query))
            response_audio_file = "audio_response.mp3"
            text_to_audio(client, response, response_audio_file)
            autoplay_audio(response_audio_file)
            os.remove(response_audio_file)
            st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()





