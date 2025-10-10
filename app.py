import streamlit as st
import joblib
import pandas as pd

# --- Configuration and Model Loading ---
# Define the filenames of the saved files
model_filename = 'case_classify_lr.pkl'
vectorizer_filename = 'tfidf_vectorizer_optimized.pkl'
# Specify the path to your model file.
MODEL_PATH = model_filename
VECTORIZER_PATH = vectorizer_filename


@st.cache_resource
def load_resource(path, resource_name):
    """Loads the model or vectorizer from a .pkl file using joblib."""
    if path is None:
        # This handles the case where the vectorizer is None (i.e., it's in a Pipeline)
        return None
    try:
        resource = joblib.load(path)
        return resource
    except FileNotFoundError:
        st.error(f"Error: {resource_name} file not found at {path}. Please check the file path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading {resource_name}: {e}")
        st.stop()


model = load_resource(MODEL_PATH, "Model")
vectorizer = load_resource(VECTORIZER_PATH, "Vectorizer")


# --- Custom CSS Styling ---

def local_css():
    """Applies custom CSS styles to make the app look prettier."""
    st.markdown(
        f"""
        <style>
            /* 1. Header Styling */
            h1 {{
                color: #FF4B4B; /* Streamlit Red */
                font-family: 'Arial Black', sans-serif;
                border-bottom: 3px solid #FF4B4B;
                padding-bottom: 10px;
            }}

            /* 2. Text Area Styling */
            /* Targeting the specific element that wraps the text area */
            textarea {{
                border: 2px solid #00BFFF; /* Deep Sky Blue border */
                border-radius: 8px;
                padding: 10px;
                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
            }}

            /* 3. Primary Button Styling ("Predict Outcome") */
            .stButton > button[kind="primary"] {{
                background-color: #00BFFF;
                color: white;
                font-weight: bold;
                border-radius: 12px;
                padding: 10px 24px;
                border: none;
            }}

            /* 4. Secondary Button Styling ("Rerun") */
            .stButton > button:not([kind="primary"]) {{
                background-color: #f0f2f6; /* Lighter background */
                color: #495057; /* Darker text */
                font-weight: bold;
                border-radius: 12px;
                padding: 10px 24px;
            }}

        </style>
        """,
        unsafe_allow_html=True
    )


local_css()  # Apply the custom styles

# --- Streamlit UI and Prediction Logic ---

st.title("Text Classification Predictor 🏷️")
st.markdown("Enter a sentence or paragraph (up to 2000 words) to get a single-word outcome prediction.")

# 1. User Input Widget
user_input_text = st.text_area(
    label="Input Text (Max 2000 words)",
    height=300,
    placeholder="Type or paste your text here...",
    key="text_input_area"
)

# Container for Buttons - MUST ONLY CREATE BUTTONS ONCE
col1, col2 = st.columns([1, 1])

with col1:
    # Use key for unique ID, set type="primary" for the custom styling
    predict_clicked = st.button("Predict Outcome", type="primary", key="predict_btn")

with col2:
    # Key is required for unique ID
    if st.button("Start New Prediction / Rerun", key="rerun_btn"):
        st.rerun()  # Forces the app to restart, clearing all inputs and results

# --- Prediction Logic Execution ---

# 2. Trigger Prediction when the button is clicked
if predict_clicked:

    # Simple validation check (optional)
    if len(user_input_text.split()) > 2000:
        st.warning("The input text seems to exceed 2000 words. Prediction might be less accurate.")

    # Check for prerequisites before proceeding
    if not user_input_text.strip():
        st.error("Please enter some text to get a prediction.")
        st.stop()
    if model is None:
        st.error("Model failed to load. Cannot proceed with prediction.")
        st.stop()

    try:
        # 3. Prepare the Input Data (FIX for "Expected 2D array" error)
        # Wrap the single text string in a list to represent one sample
        input_data_list = [user_input_text]

        # 4. Vectorization / Pipeline preparation
        if vectorizer is not None:
            # If the vectorizer is separate, transform the raw text now
            input_data_vectorized = vectorizer.transform(input_data_list)
            data_for_prediction = input_data_vectorized
        else:
            # If the model is a Pipeline, pass the raw text list directly
            data_for_prediction = input_data_list

        # 5. Get the Prediction
        prediction_array = model.predict(data_for_prediction)

        # 6. Extract and Display the Outcome
        outcome = str(prediction_array[0])

        st.success("✅ Prediction Complete!")
        st.metric(
            label="Predicted Outcome",
            value=f"**{outcome}**"
        )
        st.info("The input used was successfully processed.")

    except Exception as e:
        st.error(f"An error occurred during prediction. Check your preprocessing steps. Error: {e}")
