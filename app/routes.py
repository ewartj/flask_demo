# routes.py
from app import app
from flask import render_template, flash, redirect, url_for

import random
import pandas as pd
from app.form import Feedback_form
import requests
from flask import request


@app.route("/")
@app.route("/index", methods=['GET', 'POST'])
def index():
    form = Feedback_form()
    if request.method == 'POST':
        print("Hiiiiii")
        print(form.query.data)
        flash('Form submitted successfully!')
        print("Hiiiiii")
        data = str(form.query.data)
        print(data)
        try:
            dict1 = {
                "text": form.query.data,
                "theme": random.choice(["Theme1", "Theme2", "Theme3"]),
                "sentiment": random.choice(["Positive", "Negative", "Neutral"]),
                "theme_score": random.uniform(0, 1),
                "sentiment_score": random.uniform(0, 1)
            }
            # Create a DataFrame from the dictionary
            output_data = pd.DataFrame([dict1])
            # Save the DataFrame to a CSV file
            print(output_data)
            output_data.to_csv("db.csv", index=False)
            return redirect(url_for('results'))
        except:
            return redirect(url_for('none'))
    return render_template('index.html', form=form)

@app.route("/results", methods=['GET', 'POST'])
def results():
    output_data = pd.read_csv("db.csv")
    output_json = output_data.to_dict(orient='records')
    output_json = output_json[0]
    return render_template("bootstrap_results.html", result = output_json)

@app.route("/none", methods=['GET', 'POST'])
def none():
    return render_template("none.html")

# @app.route("/db", methods=['GET', 'POST'])
# def result_db():
#     print("result_db")
#     path = f"{getcwd()}/db.csv"
#     print(path)
#     output_data = pd.read_csv(path)
#     print(output_data)
#     output_json = output_data.to_json(orient='table',index=False)
#     print(output_json)
#     return output_json

