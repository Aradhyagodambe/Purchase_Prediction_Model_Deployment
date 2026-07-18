from flask import Flask, render_template, request
import pickle
import pandas as pd
import plotly.express as px

# Initialize Flask app (this is the 'app' in 'gunicorn app:app')
app = Flask(__name__)

# 1. Load the pre-trained Gaussian Naive Bayes model
try:
    with open('Purchase_Prediction.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: 'Purchase_Prediction.pkl' not found. Please ensure the model file is uploaded to the root directory.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    graph_html = None
    alert_type = None

    if request.method == 'POST' and model is not None:
        # 3. Input Fields processing
        gender_input = request.form.get('gender')
        gender_val = 1 if gender_input == "Male" else 0 
        
        age_val = int(request.form.get('age', 30))
        salary_val = int(request.form.get('salary', 50000))
        
        # Structure input exactly as the model expects: Gender, Age, EstimatedSalary
        input_data = pd.DataFrame(
            [[gender_val, age_val, salary_val]], 
            columns=["Gender", "Age", "EstimatedSalary"]
        )
        
        # Get probabilities for Class 0 (No Purchase) and Class 1 (Purchase)
        probabilities = model.predict_proba(input_data)[0]
        
        # Prepare data for Plotly graph
        prob_df = pd.DataFrame({
            "Outcome": ["Will Not Purchase ❌", "Will Purchase 🛍️"],
            "Probability": [probabilities[0], probabilities[1]]
        })
        
        # Create an interactive horizontal bar chart (matching your old Streamlit theme)
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
        
        # Convert the Plotly figure to HTML so Flask can display it
        graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Actionable Business Logic based on threshold
        if probabilities[1] >= 0.5:
            alert_type = "success"
        else:
            alert_type = "warning"

    return render_template('index.html', graph_html=graph_html, alert_type=alert_type)

if __name__ == "__main__":
    app.run(debug=True)
