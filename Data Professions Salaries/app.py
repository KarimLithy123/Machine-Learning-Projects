from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    sex = request.form['sex']
    designation = request.form['designation']
    age = float(request.form['age'])
    unit = request.form['unit']
    leaves_used = float(request.form['leaves_used'])
    ratings = float(request.form['ratings'])
    past_exp = float(request.form['past_exp'])
    years_worked = float(request.form['years_worked'])

    # Creating a DataFrame with the input values
    input_data = {
        'SEX': [sex],
        'DESIGNATION': [designation],
        'AGE': [age],
        'UNIT': [unit],
        'LEAVES USED': [leaves_used],
        'RATINGS': [ratings],
        'PAST EXP': [past_exp],
        'YEARS WORKED': [years_worked]
    }

    input_df = pd.DataFrame(input_data)

    # Making a prediction using the model
    prediction = model.predict(input_df)

    return render_template("index.html", prediction_text="The predicted salary is {}".format(prediction[0]))

if __name__ == "__main__":
    flask_app.run(debug=True)
