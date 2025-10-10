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


# Load the model and, potentially, the vectorizer
model = load_resource(MODEL_PATH, "Model")
vectorizer = load_resource(VECTORIZER_PATH, "Vectorizer")

# --- Streamlit UI and Prediction Logic ---

st.title("Text Classification Predictor 🏷️")
st.markdown("Enter a sentence or paragraph (up to 2000 words) to get a single-word outcome prediction.")

# 1. User Input Widget
user_input_text = st.text_area(
    label="Input Text (Max 2000 words)",
    height=300,
    placeholder="Type or paste your text here..."
)

# Simple validation check (optional)
if len(user_input_text.split()) > 2000:
    st.warning("The input text seems to exceed 2000 words. Prediction might be less accurate.")

if st.button("Predict Outcome"):

    # Check for prerequisites
    if not user_input_text.strip():
        st.error("Please enter some text to get a prediction.")
        st.stop()
    if model is None:
        st.error("Model failed to load. Cannot proceed with prediction.")
        st.stop()

    try:
        # 2. Prepare the Input Data
        # For text prediction, the model or pipeline MUST receive a list of strings,
        # even for a single input. This ensures the output is a 2D array after vectorization.
        input_data_list = [user_input_text]

        # 3. Vectorization (Only required if the model is NOT a scikit-learn Pipeline)
        if vectorizer is not None:
            # If the vectorizer is separate, transform the raw text now
            # The result is a sparse matrix, which is a 2D array representation.
            input_data_vectorized = vectorizer.transform(input_data_list)
            data_for_prediction = input_data_vectorized
        else:
            # If the model is a Pipeline, pass the raw text list directly
            data_for_prediction = input_data_list

        # 4. Get the Prediction
        # Pass the correctly formatted data to the model
        prediction_array = model.predict(data_for_prediction)

        # Extract the single word/label result
        # Ensure it's converted to a standard string for display
        outcome = str(prediction_array[0])

        # 5. Display the Outcome
        st.success("✅ Prediction Complete!")
        st.metric(
            label="Predicted Outcome",
            value=f"**{outcome}**"
        )
        st.info("The input used was successfully processed.")

    except Exception as e:
        # Display the specific error for better debugging
        st.error(f"An error occurred during prediction. Check your preprocessing steps. Error: {e}")