
import sklearn.ensemble
import sklearn.tree
import sys

# Fix old sklearn model compatibility
sys.modules['sklearn.ensemble.forest'] = sklearn.ensemble._forest
sys.modules['sklearn.tree.tree'] = sklearn.tree._classes

from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------- SAFE INPUT HANDLER ----------------
def get_numeric_inputs(form_data):
    values = []
    for key in form_data:
        val = form_data.get(key)
        if val is None or val.strip() == "":
            continue
        try:
            values.append(float(val))
        except:
            continue
    return values


# ---------------- MODEL PREDICT ----------------
def predict(values):
    values = np.asarray(values)

    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl','rb'))
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
    else:
        return "Invalid input length"

    return model.predict(values.reshape(1, -1))[0]


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template('home.html')


@app.route("/diabetes")
def diabetesPage():
    return render_template('diabetes.html')


@app.route("/cancer")
def cancerPage():
    return render_template('breast_cancer.html')


@app.route("/heart")
def heartPage():
    return render_template('heart.html')


@app.route("/kidney")
def kidneyPage():
    return render_template('kidney.html')


@app.route("/liver")
def liverPage():
    return render_template('liver.html')


@app.route("/malaria")
def malariaPage():
    return render_template('malaria.html')


@app.route("/pneumonia")
def pneumoniaPage():
    return render_template('pneumonia.html')


# ---------------- MAIN PREDICT ----------------
@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        values = get_numeric_inputs(request.form)

        if len(values) == 0:
            return render_template("home.html", message="Please enter valid data")

        pred = predict(values)

    except Exception as e:
        return render_template("home.html", message=f"Error: {str(e)}")

    return render_template('predict.html', pred=pred)


# ---------------- MALARIA ----------------
@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():
    try:
        if 'image' not in request.files:
            return render_template('malaria.html', message="Upload Image")

        img = Image.open(request.files['image'])
        img = img.resize((36,36))
        img = np.asarray(img)
        img = img.reshape((1,36,36,3))
        img = img.astype(np.float64)

        model = load_model("models/malaria.h5")
        pred = np.argmax(model.predict(img)[0])

    except Exception as e:
        return render_template('malaria.html', message=str(e))

    return render_template('malaria_predict.html', pred=pred)


# ---------------- PNEUMONIA ----------------
@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():
    try:
        if 'image' not in request.files:
            return render_template('pneumonia.html', message="Upload Image")

        img = Image.open(request.files['image']).convert('L')
        img = img.resize((36,36))
        img = np.asarray(img)
        img = img.reshape((1,36,36,1))
        img = img / 255.0

        model = load_model("models/pneumonia.h5")
        pred = np.argmax(model.predict(img)[0])

    except Exception as e:
        return render_template('pneumonia.html', message=str(e))

    return render_template('pneumonia_predict.html', pred=pred)


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)