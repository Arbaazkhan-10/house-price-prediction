
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("house_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Receive user input
    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    age = int(request.form["age"])

    # Convert inputs into DataFrame
    input_data = pd.DataFrame({
        "area": [area],
        "bedroom": [bedrooms],
        "bathrooms": [bathrooms],
        "age": [age]
    })

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Return result
    return render_template(
    "result.html",
    prediction=f"{prediction:,.0f}"
    )


if __name__ == "__main__":
    app.run(debug=True)
