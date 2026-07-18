import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# 1. Load the pre-trained Gaussian Naive Bayes model
@st.cache_resource
def load_model():
    # Make sure your original model is saved as 'model.pkl' in the same directory
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please ensure the model file is uploaded to the root directory.")
    st.stop()

# 2. UI Setup: Shopping Mart Theme
st.set_page_config(page_title="SmartMart Predictor", page_icon="🛒", layout="centered")

st.title("🛒 SmartMart Customer Insights 🛍️")
st.markdown("""
Welcome to the **SmartMart Purchase Predictor**! 
Enter the customer's profile below to calculate their likelihood of checking out with items today.
""")

st.divider()

# 3. Input Fields
st.subheader("👤 Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    # Assuming Gender was label encoded as 0 and 1 during training
    gender_input = st.selectbox("Gender", ["Female", "Male"])
    gender_val = 1 if gender_input == "Male" else 0 

with col2:
    age_val = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

with col3:
    salary_val = st.number_input("Estimated Salary ($)", min_value=0, max_value=500000, value=50000, step=5000)

st.divider()

# 4. Prediction and Graph
if st.button("Predict Purchase Probability 🚀", use_container_width=True):
    
    # Structure input exactly as the model expects: Gender, Age, EstimatedSalary
    input_data = pd.DataFrame(
        [[gender_val, age_val, salary_val]], 
        columns=["Gender", "Age", "EstimatedSalary"]
    )
    
    # Get probabilities for Class 0 (No Purchase) and Class 1 (Purchase)
    probabilities = model.predict_proba(input_data)[0]
    
    st.markdown("### 📊 Checkout Prediction")
    
    # Prepare data for Plotly graph
    prob_df = pd.DataFrame({
        "Outcome": ["Will Not Purchase ❌", "Will Purchase 🛍️"],
        "Probability": [probabilities[0], probabilities[1]]
    })
    
    # Create an interactive horizontal bar chart
    fig = px.bar(
        prob_df,
        x="Probability",
        y="Outcome",
        orientation='h',
        color="Outcome",
        color_discrete_map={
            "Will Not Purchase ❌": "#FF6B6B", 
            "Will Purchase 🛍️": "#4ECDC4"
        },
        text_auto='.1%'
    )
    
    fig.update_layout(
        xaxis_tickformat='.0%', 
        xaxis_title="Likelihood", 
        yaxis_title="",
        showlegend=False,
        height=250,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Actionable Business Logic based on threshold
    if probabilities[1] >= 0.5:
        st.success("**High Intent Customer!** Recommend premium items at checkout or offer a loyalty card. 🌟")
    else:
        st.warning("**Low Intent Customer.** Consider offering a time-sensitive discount coupon to encourage conversion! 🏷️")
