from flask import Flask, request, render_template
from src.pipline.prediction_pipline import PredictPipline, CustomData
import pandas as pd
import numpy as np
import os

application = Flask(__name__)
app = application

# Helper function to format rupees
def format_inr(value):
    if value >= 1_00_00_000:
        return f"₹{value / 1_00_00_000:.2f} Cr"
    elif value >= 1_00_000:
        return f"₹{value / 1_00_000:.2f} Lakh"
    else:
        return f"₹{value:,.0f}"

@app.route("/", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        csv_data = pd.read_csv(os.path.join("../MUMBAI HOUSE PRICE PREDICTOR/artifcats", "raw.csv"))
        return render_template("form.html", Location=csv_data["Location"].unique())

    else:
        data = CustomData(
            Area=int(request.form.get("Area")),
            Location=request.form.get("Location"),
            No_of_Bedrooms=float(request.form.get("No_of_Bedrooms")),
            New_Resale=int(request.form.get("New_Resale")),
            Gymnasium=int(request.form.get("Gymnasium")),
            Lift_Available=int(request.form.get("Lift_Available")),
            Car_Parking=int(request.form.get("Car_Parking")),
            Maintenance_Staff=float(request.form.get("Maintenance_Staff")),
            _24x7_Security=int(request.form.get("_24x7_Security")),
            Childrens_Play_Area=int(request.form.get("Childrens_Play_Area")),
            Clubhouse=int(request.form.get("Clubhouse")),
            Intercom=int(request.form.get("Intercom")),
            Landscaped_Gardens=int(request.form.get("Landscaped_Gardens")),
            Indoor_Games=int(request.form.get("Indoor_Games")),
            Gas_Connection=int(request.form.get("Gas_Connection")),
            Jogging_Track=int(request.form.get("Jogging_Track")),
            Swimming_Pool=int(request.form.get("Swimming_Pool"))
        )

        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.Predict(final_data)

        # Handle log-transformed output if applicable
        # Uncomment the line below if you trained on log1p(price)
        # result = np.expm1(pred[0])

        # If not log-transformed, use raw prediction
        result = pred[0]
        formatted_result = format_inr(result)

        csv_data = pd.read_csv(os.path.join("../MUMBAI HOUSE PRICE PREDICTOR/artifcats", "raw.csv"))
        return render_template("form.html", final_result=f"Your House Price Is: {formatted_result}",
                               Location=csv_data["Location"].unique())

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
