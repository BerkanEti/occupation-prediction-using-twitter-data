
import streamlit as st
import os
from model_prediction import OccupationPredictorML, OccupationPredictorDL
import plotly.graph_objects as go

st.set_page_config(page_title="Turkish Occupation Predictor", layout="wide")
st.markdown("""
    <style>
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)
    
def value_to_color(value):
    """
    Converts a value in the range [0, 100] to a color in hexadecimal RGB format.

    :param value: A number in the range [0, 100]
    :return: A string representing the color in '#RRGGBB' format
    """
    if not (0 <= value <= 100):
        raise ValueError("Value must be in the range [0, 100]")

    red = 0
    green = int(255 * (1 - value / 100))
    blue = 255

    return f"#{red:02x}{green:02x}{blue:02x}"

def create_probability_chart(probabilities, title):
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    occupations, values = zip(*sorted_items)

    colors = [value_to_color(value) for value in values]

    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=occupations,
            orientation='h',
            marker_color=colors,
            text=[f'%{v:.1f}' for v in values],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Probability (%)",
        yaxis_title="Occupation",
        height=max(400, len(occupations) * 30),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def main():
    st.title("Turkish Occupation Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            options=["ML", "DL"],
            help="Choose between Machine Learning and Deep Learning models"
        )
    
    with col2:
        lemmatization = st.selectbox(
            "Lemmatization Method",
            options=["zeyrek", "zemberek"],
            help="Choose the lemmatization method for text processing"
        )
    
    with col3:
        tweet_count_display = st.selectbox(
            "Tweet Combination Size",
            options=[1, 2, 3, 5],
            key="tweet_count_display",
            help="Select the number of tweets used for training"
        )
        tweet_count = f"{tweet_count_display}_tweets"
    
    model_path = f"saved_models/trained_{lemmatization}_{tweet_count}/{model_type.lower()}/"
    
    st.code(model_path, language="bash")
    
    user_input = st.text_area(
        "Enter Turkish Text",
        height=100,
        help="Enter the text you want to analyze",
        placeholder="Enter text in Turkish for prediction..."
    )
    
    if st.button("Predict", type="primary", disabled=not user_input):
        try:
            with st.spinner("Processing..."):
                predictor = OccupationPredictorML(model_path) if model_type == "ML" else OccupationPredictorDL(model_path)
                
                results = predictor.predict(user_input)
        
                st.subheader("Prediction Results")
                
                for model_name, result in results.items():
                    with st.expander(f"Model: {model_name}", expanded=True):
                
                        st.markdown(f"### Predicted Occupation: **{result['prediction']}**")
                    
                        if 'confidence' in result:
                            st.metric("Confidence", f"{result['confidence']:.1f}%")
                    
                        if 'probabilities' in result:
                            probabilities = {k: float(v) if isinstance(v, str) else v 
                                          for k, v in result['probabilities'].items()}
                            
                            fig = create_probability_chart(
                                probabilities,
                                f"Probability Distribution - {model_name}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()