from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from unidecode import unidecode

app = Flask(__name__)
app.model = {}
with open("static/model/saved_city_models.pkl", 'rb') as f:
    app.model = pickle.load(f)  


@app.route("/")
def welcome():
    return render_template('overlay_without_result.html')


@app.route("/cityAnalysis", methods=["POST"])
def analyzeCity():
    city_name = unidecode(str(request.form.get("city"))).lower()
    input_date = request.form.get("input_date")
    days = int((datetime.strptime(input_date, '%Y-%m-%d') - datetime.strptime('1742-01-01', '%Y-%m-%d')).days)
    predicted_value = str(app.model[city_name].predict(PolynomialFeatures(degree=3).fit_transform(pd.Series([days]).values.reshape(-1, 1)))[0])
    return render_template('overlay_with_result.html', city_name_unmodified=request.form.get("city"), city_name=city_name, input_date=input_date, predicted_value=round(float(predicted_value),2), img_address="static/plot_imgs/"+city_name+".jpg")


if __name__=="__main__":
    app.run(host='0.0.0.0', port=8080)
