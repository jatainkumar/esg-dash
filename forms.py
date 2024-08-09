from flask_wtf import FlaskForm
from wtforms import (
    IntegerField,
    BooleanField,
    SubmitField
)
from wtforms.validators import DataRequired, Optional

class InputForm(FlaskForm):
    national_inv = IntegerField(
        label="National Inventory *",
        validators=[DataRequired()]
    )
    lead_time = IntegerField(
        label="Lead Time *",
        validators=[DataRequired()]
    )
    in_transit_qty = IntegerField(
        label="In Transit Qty (Optional)",
        validators=[Optional()]
    )
    forecast_3_month = IntegerField(
        label="Forecast 3 Month *",
        validators=[DataRequired()]
    )
    forecast_6_month = IntegerField(
        label="Forecast 6 month (Optional)",
        validators=[Optional()]
    )
    forecast_9_month = IntegerField(
        label="Forecast 9 month (Optional)",
        validators=[Optional()]
    )
    sales_1_month = IntegerField(
        label="Sales 1 Month (Optional)",
        validators=[Optional()]
    )
    sales_3_month = IntegerField(
        label="Sales 3 Month (Optional)",
        validators=[Optional()]
    )
    sales_6_month = IntegerField(
        label="Sales 6 Month (Optional)",
        validators=[Optional()]
    )
    sales_9_month = IntegerField(
        label="Sales 9 Month (Optional)",
        validators=[Optional()]
    )
    min_bank = IntegerField(
        label="Minimum Bank (Optional)",
        validators=[Optional()]
    )
    potential_issue = IntegerField(
        label="Potential issue (Optional)",
        validators=[Optional()]
    )
    pieces_past_due = IntegerField(
        label="Pieces past due (Optional)",
        validators=[Optional()]
    )
    perf_6_month_avg = IntegerField(
        label="Performance 6 Month Avg (Optional)",
        validators=[Optional()]
    )
    perf_12_month_avg = IntegerField(
        label="Performance 12 Month Avg (Optional)",
        validators=[Optional()]
    )
    local_bo_qty = IntegerField(
        label="Local Backorder Quantity (Optional)",
        validators=[Optional()]
    )
    deck_risk = IntegerField(
        label="Deck Risk (Optional)",
        validators=[Optional()]
    )
    oe_constraint = IntegerField(
        label="OE Constraint (Optional)",
        validators=[Optional()]
    )
    ppap_risk = IntegerField(
        label="PPAP Risk (Optional)",
        validators=[Optional()]
    )
    stop_auto_buy = IntegerField(
        label="Stop Auto Buy (Optional)",
        validators=[Optional()]
    )
    rev_stop = IntegerField(
        label="Rev Stop (Optional)",
        validators=[Optional()]
    )
    num_trucks = IntegerField(
        label="Number of Trucks *",
        validators=[DataRequired()]
    )
    submit = SubmitField("Predict")
