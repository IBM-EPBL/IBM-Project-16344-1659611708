import joblib
from flask import Flask, render_template,request,redirect,url_for
import numpy as np
import pandas as pd

df = pd.read_csv("Dataset/crop_production.csv")
data = df.dropna()
data = data.drop(["District_Name","Crop_Year"],axis=1)
sum_maxp = data["Production"].sum()
data["percent_of_production"] = data["Production"].map(lambda x:(x/sum_maxp)*100)
data1 = data.drop('Production',axis=1)
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['email'] != 'admin@gmail.com' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        elif request.form['email'] == '' or request.form['password'] == '':
            error='Please fill the form'
        else:
            return redirect(url_for('dashboard'))
    return render_template('login.html',error=error)


@app.route('/register',methods=['GET','POST'])
def register():
    msg=''
    if request.method == 'POST' and 'password' in request.form and 'confirmpwd' in request.form and 'email' in request.form :
        msg = 'You have successfully registered !'
        return redirect(url_for('dashboard'))
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template('register.html',msg=msg)

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, 5)
    row = pd.DataFrame(to_predict, columns =['State_Name', 'Season', 'Crop','Area','percent_of_production'])
    data2 = pd.concat([row, data1]).reset_index(drop = True)
    data2['Area'] = data2['Area'].astype(float)
    data2['percent_of_production'] = data2['percent_of_production'].astype(float)
    features = pd.get_dummies(data2)
    features = np.array(features)
    loaded_model = joblib.load('model_final.sav')
    result = loaded_model.predict(features[:1,:165])
    return result[0]

@app.route("/prediction",  methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        result = ValuePredictor(to_predict_list, 5)
        return render_template('result.html',prediction=result)
    return render_template('prediction.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/story')
def story():
    return render_template('story.html')