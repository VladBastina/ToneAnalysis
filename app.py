import streamlit as st
from streamlit_mic_recorder import mic_recorder
import google.generativeai as genai
import plotly.express as px
import pandas as pd
import os
import io
import tempfile
import json
from pydub import AudioSegment # Used to ensure WAV format if needed

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Audio Sentiment Analysis")
st.title("🗣️ Audio Sentiment Analysis with Gemini")
st.markdown("""
Upload a WAV file, record new audio, or use the default example. The app will use Google's Gemini model
to analyze the sentiment, focusing on the customer if it detects a support call.
""")

# --- Default File Configuration ---
DEFAULT_AUDIO_FILENAME = "default_audio.wav" # MAKE SURE THIS FILE EXISTS!

# --- API Key Handling ---
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
if not api_key:
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")

if not api_key:
    st.warning("Please enter your Gemini API Key to proceed.")
    st.stop()

try:
    genai.configure(api_key=api_key)
    # Use a model that supports audio input, like 1.5 Flash or 1.5 Pro
    model = genai.GenerativeModel(model_name="gemini-2.5-pro-exp-03-25") # Or gemini-1.5-pro
except Exception as e:
    st.error(f"Error configuring Gemini SDK: {e}")
    st.stop()


st.sidebar.image("zega_logo.PNG",use_container_width=True)
# --- Function Definitions (Keep analyze_audio, detailed_sentiment_prompt, plot_sentiment_timeline as before) ---

def detailed_sentiment_prompt(is_customer_support=None, customer_focus=False):
    """Generates the prompt for Gemini based on context."""
    base_prompt = """
    Analyze the sentiment of the provided audio conversation in detail. Consider the following aspects:
    1.  **Voice Tone:** (e.g., calm, agitated, happy, sad, sarcastic, urgent, monotone)
    2.  **Voice Intensity:** (e.g., loud, quiet, normal, shouting, whispering)
    3.  **Speaking Pace:** (e.g., fast, slow, normal, rushed, hesitant)
    4.  **Specific Emotions:** Identify primary emotions expressed (e.g., frustration, relief, anger, confusion, satisfaction, politeness, impatience).

    First, determine if this sounds like a customer support interaction (e.g., someone calling a company for help). Respond 'Customer Support: Yes' or 'Customer Support: No'.

    """

    if is_customer_support is None: # Initial analysis phase
         prompt = base_prompt + """
    Based on your determination above, proceed with the sentiment analysis.

    **Sentiment Timeline:** Provide a timeline of the overall sentiment throughout the conversation. Divide the audio into logical segments (e.g., every 15-20 seconds or by speaker turn if discernible) and assign a sentiment score from -10 (very negative) to +10 (very positive) for each segment.

    **Output Format:** Structure your entire response strictly as a JSON object with the following keys:
    - "is_customer_support": (boolean, true if it's customer support, false otherwise)
    - "analysis_target": (string, "customer only" or "full conversation")
    - "detailed_report": (string, a comprehensive text report covering tone, intensity, pace, emotions, and overall sentiment trends based on the analysis target)
    - "sentiment_timeline": (array of numbers, e.g., [2, 1, -5, -3, 0, 4, 6])
    """

    elif is_customer_support and customer_focus:
        prompt = base_prompt + """
    **Focus:** Since this is identified as a customer support call, focus your analysis *exclusively* on the speech segments likely belonging to the **customer**. Ignore the agent's speech for sentiment scoring and detailed analysis unless it directly influences the customer's reaction.

    **Sentiment Timeline:** Provide a timeline of the **customer's** sentiment throughout the conversation. Divide the customer's speaking parts into logical segments and assign a sentiment score from -10 (very negative) to +10 (very positive) for each segment.

    **Output Format:** Structure your entire response strictly as a JSON object with the following keys:
    - "is_customer_support": true
    - "analysis_target": "customer only"
    - "detailed_report": (string, a comprehensive text report covering the *customer's* tone, intensity, pace, emotions, and overall sentiment trends)
    - "sentiment_timeline": (array of numbers, representing the *customer's* sentiment scores, e.g., [-5, -6, -2, 1, 5])
    """
    else: # Not customer support, or explicitly analyze full conversation
         prompt = base_prompt + """
    **Focus:** Analyze the sentiment of the **entire conversation**, considering all speakers.

    **Sentiment Timeline:** Provide a timeline of the overall sentiment throughout the conversation. Divide the audio into logical segments (e.g., every 15-20 seconds or by speaker turn) and assign a sentiment score from -10 (very negative) to +10 (very positive) for each segment.

    **Output Format:** Structure your entire response strictly as a JSON object with the following keys:
    - "is_customer_support": false
    - "analysis_target": "full conversation"
    - "detailed_report": (string, a comprehensive text report covering tone, intensity, pace, emotions, and overall sentiment trends for the *whole conversation*)
    - "sentiment_timeline": (array of numbers, representing the *overall* sentiment scores, e.g., [2, 1, -5, -3, 0, 4, 6])
    """
    return prompt


def analyze_audio(audio_bytes, filename="uploaded_audio.wav"):
    """Sends audio to Gemini and processes the response."""
    temp_file_path = None
    uploaded_file_info = None
    try:
        # Gemini SDK works best with files. Save bytes to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(audio_bytes)
            temp_file_path = tmpfile.name

        # Optional: Ensure it's WAV format for robustness
        # try:
        #     audio_segment = AudioSegment.from_file(temp_file_path)
        #     audio_segment.export(temp_file_path, format="wav") # Re-export as WAV
        # except Exception as e:
        #     st.warning(f"Could not verify/re-export as WAV using pydub: {e}. Sending as is.")


        # Upload the file to Gemini
        uploaded_file_info = genai.upload_file(path=temp_file_path, mime_type="audio/wav")
    
        # --- Initial Analysis Phase (Determine if Customer Support) ---
        initial_prompt = detailed_sentiment_prompt()
        initial_response = model.generate_content([initial_prompt, uploaded_file_info],
                                                   request_options={"timeout": 600}) # Increased timeout

        # --- Process Initial Response ---
        try:
            # Clean potential markdown/code block formatting
            cleaned_text = initial_response.text.strip().lstrip('```json').rstrip('```')
            initial_data = json.loads(cleaned_text)
            is_customer_support = initial_data.get("is_customer_support", False)

            # --- Second Analysis Phase (Refined based on support type) ---
            # Decide if we need a second pass to focus on the customer
            needs_second_pass = is_customer_support
            if needs_second_pass:
                refined_prompt = detailed_sentiment_prompt(is_customer_support=True, customer_focus=True)
                final_response = model.generate_content([refined_prompt, uploaded_file_info],
                                                        request_options={"timeout": 600})
                final_text = final_response.text.strip().lstrip('```json').rstrip('```')
                analysis_data = json.loads(final_text)
            else:
                # Use the results from the first pass if not customer support
                analysis_data = initial_data # Reuse initial analysis

            # Validate keys exist
            report = analysis_data.get("detailed_report", "Report not found in response.")
            timeline = analysis_data.get("sentiment_timeline", [])
            target = analysis_data.get("analysis_target", "unknown")

            return report, timeline, target, is_customer_support

        except json.JSONDecodeError:
            st.error("Error: Could not parse Gemini's response as JSON. Raw response:")
            st.code(initial_response.text if 'initial_response' in locals() else "No initial response captured")
            if 'final_response' in locals():
                st.code(final_response.text)
            return "Error parsing response.", [], "Error", False
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            return f"Error: {e}", [], "Error", False

    except Exception as e:
        st.error(f"An error occurred during file processing or API call: {e}")
        return f"Error: {e}", [], "Error", False
    finally:
        # Clean up the uploaded file on Gemini and the local temp file
        if uploaded_file_info:
            try:
                # Gemini API might change; adapt if delete() method isn't available
                # print(f"Attempting to delete file: {uploaded_file_info.name}") # Debugging
                genai.delete_file(uploaded_file_info.name)
            except AttributeError:
                 st.warning(f"Could not directly delete file object. Attempting delete by name: {uploaded_file_info.name}")
                 try:
                     genai.delete_file(uploaded_file_info.name)
                 except Exception as del_err_name:
                      st.warning(f"Could not delete uploaded file from Gemini by name either: {del_err_name}")
            except Exception as del_err:
                st.warning(f"Could not delete uploaded file from Gemini: {del_err}")
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def plot_sentiment_timeline(timeline_data):
    """Generates a Plotly line chart for the sentiment timeline."""
    if not timeline_data or not isinstance(timeline_data, list):
        st.warning("No valid sentiment timeline data to plot.")
        return None

    # Ensure data are numbers (handle potential strings if parsing failed slightly)
    numeric_timeline = []
    for item in timeline_data:
        try:
            numeric_timeline.append(float(item))
        except (ValueError, TypeError):
            st.warning(f"Skipping non-numeric value in timeline: {item}")
            # Optionally append a neutral value like 0 or None, or just skip
            # numeric_timeline.append(0)

    if not numeric_timeline:
        st.warning("No numeric sentiment data available after filtering.")
        return None

    df = pd.DataFrame({
        'Segment': range(1, len(numeric_timeline) + 1),
        'Sentiment Score': numeric_timeline
    })

    fig = px.line(df, x='Segment', y='Sentiment Score',
                  title="Sentiment Progression Over Conversation Segments",
                  markers=True, range_y=[-10.5, 10.5]) # Set Y-axis range
    fig.update_layout(xaxis_title="Conversation Segment / Time Progression",
                      yaxis_title="Sentiment Score (-10 to +10)")
    return fig


# --- Streamlit UI Elements ---
audio_bytes = None
file_name = None

# --- ADDED "Use Default Example" option ---
input_method = st.radio(
    "Choose audio input method:",
    ("Upload WAV file", "Record Audio", "Use Default Example (Customer support call)"),
    index=0,
    key="input_method"
)

if input_method == "Upload WAV file":
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'], key="uploader")
    if uploaded_file is not None:
        file_name = uploaded_file.name
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes, format='audio/wav')

elif input_method == "Record Audio":
    st.write("Click the microphone to start/stop recording (allow microphone access).")
    # Use streamlit_mic_recorder
    # The key='recorder' helps maintain state across reruns
    audio_info = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False, # Allow multiple recordings without refresh
        use_container_width=True,
        format="wav", # Specify wav format
        key='recorder'
    )

    if audio_info and audio_info['bytes']:
        st.success("Recording finished!")
        audio_bytes = audio_info['bytes']
        file_name = "recorded_audio.wav"
        st.audio(audio_bytes, format='audio/wav')
        # Optional: ensure WAV format integrity if needed
        # try:
        #     audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        #     wav_buffer = io.BytesIO()
        #     audio_segment.export(wav_buffer, format="wav")
        #     audio_bytes = wav_buffer.getvalue()
        #     st.info("Ensured audio is in WAV format.")
        # except Exception as e:
        #     st.warning(f"Could not process recorded audio with pydub: {e}. Sending as is.")


# --- ADDED Logic for Default Example ---
elif input_method == "Use Default Example (Customer support call)":
    default_file_path = DEFAULT_AUDIO_FILENAME
    # Check if the default file exists in the script's directory
    if os.path.exists(default_file_path):
        st.info(f"Using default example file: '{default_file_path}'")
        try:
            with open(default_file_path, "rb") as f:
                audio_bytes = f.read()
            file_name = os.path.basename(default_file_path)
            # Display the audio player for the default file
            st.audio(audio_bytes, format='audio/wav')
        except Exception as e:
            st.error(f"Error reading default file '{default_file_path}': {e}")
            audio_bytes = None # Reset to prevent analysis button
            file_name = None
    else:
        # Handle case where the file is missing
        st.error(f"Default file not found: '{default_file_path}'.")
        st.markdown(f"Please make sure a file named `{DEFAULT_AUDIO_FILENAME}` exists in the same directory as the Streamlit script (`app.py`).")
        # Ensure analysis button doesn't appear if file is missing
        audio_bytes = None
        file_name = None


# --- Analysis Trigger ---
# This part remains the same, it checks if audio_bytes and file_name are set,
# regardless of how they were set (upload, record, or default)
if audio_bytes and file_name:
    if st.button(f"Analyze Sentiment for '{file_name}'", key="analyze_button"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Sentiment Analysis Report")
            with st.spinner("Analyzing audio... This may take a minute or two depending on length."):
                report, timeline, target, is_cs = analyze_audio(audio_bytes, file_name)

            st.text_area("Detailed Report", report, height=400)

        with col2:
            st.subheader("📈 Sentiment Timeline Plot")
            if timeline:
                fig = plot_sentiment_timeline(timeline)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not generate plot.")
            else:
                st.info("No sentiment timeline data available to plot.")
# Don't show the button instruction if using default and file is missing
elif input_method != "Use Default Example" or os.path.exists(DEFAULT_AUDIO_FILENAME) :
     st.info("Please provide audio via one of the methods above to begin analysis.")

# --- Footer/Info ---
st.markdown("---")
st.markdown("Powered by ZEGA AI")