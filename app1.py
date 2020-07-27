import joblib
from flask import Flask, render_template, request

# the model is loaded

spam_model = open('spam_model.pkl', 'rb')
clf = joblib.load(spam_model)


# app
app = Flask(__name__)


@app.route('/')  # Routes the app to below task when the URL is called
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict_fn():
    spam_model = open('spam_model.pkl', 'rb')
    clf = joblib.load(spam_model)

    cvect_model = open('cvect.pkl', 'rb')
    cv = joblib.load(cvect_model)


    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        cvect = cv.transform(data).toarray()
        my_prediction = clf.predict(cvect)

    return render_template('result.html', prediction=my_prediction)


# Calling the main function and running the flask app

if __name__ == '__main__':
    app.run(debug=True)
