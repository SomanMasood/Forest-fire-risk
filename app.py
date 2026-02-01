from flask import Flask, render_template, request
import numpy as np
import pickle
import traceback

app = Flask(__name__)

# Load model and scaler
with open("artifacts/ridge.pkl", "rb") as f:
    model = pickle.load(f)

with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


def safe_float(value, field_name, min_val=None, max_val=None):
    if value is None or value == "":
        raise ValueError(f"Missing value for {field_name}")

    val = float(value)

    if min_val is not None and val < min_val:
        raise ValueError(f"{field_name} must be ≥ {min_val}")

    if max_val is not None and val > max_val:
        raise ValueError(f"{field_name} must be ≤ {max_val}")

    return val


def classify_risk(fwi):
    if fwi < 10:
        return "Low", "green"
    elif fwi < 20:
        return "Moderate", "yellow"
    elif fwi < 30:
        return "High", "orange"
    else:
        return "Extreme", "red"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    risk_label = None
    risk_color = None

    if request.method == "POST":
        try:
            features = [
    
    safe_float(request.form.get("temperature"), "Temperature", -10, 60),
    safe_float(request.form.get("rh"), "Relative Humidity", 0, 100),
    safe_float(request.form.get("wind"), "Wind Speed", 0),
    safe_float(request.form.get("rain"), "Rain", 0),
    safe_float(request.form.get("ffmc"), "FFMC", 0),
    safe_float(request.form.get("dmc"), "DMC", 0),
    safe_float(request.form.get("isi"), "ISI", 0),
    safe_float(request.form.get("classes"), "Classes", 0, 1),
    safe_float(request.form.get("region"), "Region", 0, 1),
]


            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)

            fwi = model.predict(X_scaled)[0]
            prediction = round(fwi, 2)
            risk_label, risk_color = classify_risk(fwi)

        except Exception as e:
            import traceback
            traceback.print_exc()
            prediction = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        risk_label=risk_label,
        risk_color=risk_color
    )



if __name__ == "__main__":
      app.run()

