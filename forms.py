from flask_wtf import FlaskForm
from wtforms import (
    IntegerField,
    SubmitField
)
from wtforms.validators import DataRequired

class InputForm(FlaskForm):
    national_inv = IntegerField(
        label="National Inventory",
        validators=[DataRequired()]
    )
    lead_time = IntegerField(
        label="Lead Time",
        validators=[DataRequired()]
    )
    forecast_3_month = IntegerField(
        label="Forecast 3 Month",
        validators=[DataRequired()]
    )
    num_trucks = IntegerField(
        label="Number of Trucks",
        validators=[DataRequired()]
    )
    submit = SubmitField("Predict")
