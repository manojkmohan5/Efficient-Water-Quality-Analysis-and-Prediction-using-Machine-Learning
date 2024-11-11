from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('M:\Efficient Water Quality Analysis\Project_demo\wqi.joblib')

@app.route('/')
def home():
    return render_template("web.html")

@app.route('/login', methods=['POST'])
def login():
    # Extract form data
    year = request.form["year"]
    do = request.form["do"]
    ph = request.form["ph"]
    co = request.form["co"]
    bod = request.form["bod"]
    tc = request.form["tc"]
    na = request.form["na"]

    # Prepare input for prediction
    total = [[float(do), float(ph), float(co), float(bod), float(na), float(tc)]]
    y_pred = model.predict(total)[0]  # Get the predicted value

    # Determine the category based on prediction
    if 95 <= y_pred <= 100:
        message = f'Excellent, The Predicted Value Is {y_pred}'
    elif 89 <= y_pred <= 94:
        message = f'Very Good, The Predicted Value Is {y_pred}'
    elif 80 <= y_pred <= 88:
        message = f'Good, The Predicted Value Is {y_pred}'
    elif 65 <= y_pred <= 79:
        message = f'Fair, The Predicted Value Is {y_pred}'
    elif 45 <= y_pred <= 64:
        message = f'Marginal, The Predicted Value Is {y_pred}'
    else:
        message = f'Poor, The Predicted Value Is {y_pred}'

    return render_template("web.html", showcase=message)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
