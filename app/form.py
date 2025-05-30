from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class Feedback_form(FlaskForm):
    query = StringField('query')
    submit = SubmitField('submit')