from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('saved_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    sepal_length = float(request.form['SepalLengthCm'])
    sepal_width = float(request.form['SepalWidthCm'])
    petal_length = float(request.form['PetalLengthCm'])
    petal_width = float(request.form['PetalWidthCm'])

    # Predict using the loaded model
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    result = prediction[0]  # Assuming prediction returns a single result

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)


