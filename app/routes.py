from app import app
from flask import render_template, request
from app.form import Feedback_form
from app.model import predict


@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    form = Feedback_form()
    if request.method == "POST" and form.validate_on_submit():
        text = form.query.data.strip()
        if not text:
            return render_template("index.html", form=form, error="Please enter some text.")
        try:
            result = predict(text)
            return render_template("sentiment_results.html", result=result)
        except Exception as e:
            app.logger.error(f"Prediction error: {e}")
            return render_template("index.html", form=form, error="Prediction failed. Please try again.")
    return render_template("index.html", form=form)
