from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load("best_gb_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    gender = request.form["Gender"]
    hours = float(request.form["HoursStudied"])
    tutoring = request.form["Tutoring"]
    region = request.form["Region"]
    attendance = float(request.form["Attendance"])
    parent_edu = request.form["Parent_Education"]

    # Factorize-style encoding for categorical values
    # Same mapping used during training
    mapping_gender = {"Male": 0, "Female": 1}
    mapping_tutor = {"No": 0, "Yes": 1}
    mapping_region = {"Urban": 0, "Rural": 1}
    mapping_parent = {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}

    gender_val = mapping_gender.get(gender, 0)
    tutor_val = mapping_tutor.get(tutoring, 0)
    region_val = mapping_region.get(region, 0)
    parent_val = mapping_parent.get(parent_edu, 0)

    # Prepare features
    features = np.array([[gender_val, hours, tutor_val, region_val, attendance, parent_val]])

    # Predict score
    prediction = model.predict(features)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
