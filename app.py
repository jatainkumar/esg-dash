import networkx as nx
import geopandas as gpd
import folium
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import random
from IPython.display import display
from docplex.mp.model import Model
import geodatasets
from geodatasets import get_path


import torch
import torch.nn as nn
import torch.nn.functional as F



from flask import Flask, url_for, render_template, request
from forms import InputForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret_key"

class Net(nn.Module):
    def __init__(self, in_count, output_count):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_count, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, output_count)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return self.softmax(x)
    
input_dim = 21 
model1 = Net(input_dim, 2)
model1.load_state_dict(torch.load('model.pth'))
model1.eval()



@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        input_data = {
            'national_inv': form.national_inv.data,
            'lead_time': form.lead_time.data,
            'in_transit_qty': form.in_transit_qty.data if form.in_transit_qty.data else 44.05202209,
            'forecast_3_month': form.forecast_3_month.data,
            'forecast_6_month': form.forecast_6_month.data if form.forecast_6_month.data else 344.986663585842,
            'forecast_9_month': form.forecast_9_month.data if form.forecast_9_month.data else 506.364430699228,
            'sales_1_month': form.sales_1_month.data if form.sales_1_month.data else 55.9260685127913,
            'sales_3_month': form.sales_3_month.data if form.sales_3_month.data else 175.025930468166,
            'sales_6_month': form.sales_6_month.data if form.sales_6_month.data else 341.728839477207,
            'sales_9_month': form.sales_9_month.data if form.sales_9_month.data else 525.269700686075,
            'min_bank': form.min_bank.data if form.min_bank.data else 52.7723033900916,
            'potential_issue': int(form.potential_issue.data) if form.potential_issue.data else 0.00053736652485,
            'pieces_past_due': form.pieces_past_due.data if form.pieces_past_due.data else 2.04372400554548,
            'perf_6_month_avg': form.perf_6_month_avg.data if form.perf_6_month_avg.data else -6.8720588378183,
            'perf_12_month_avg': form.perf_12_month_avg.data if form.perf_12_month_avg.data else -6.43794674321329,
            'local_bo_qty': form.local_bo_qty.data if form.local_bo_qty.data else 0.626450653490218,
            'deck_risk': int(form.deck_risk.data) if form.deck_risk.data else 0.229570444485653,
            'oe_constraint': int(form.oe_constraint.data) if form.oe_constraint.data else 0.0001451541329528,
            'ppap_risk': int(form.ppap_risk.data) if form.ppap_risk.data else 0.120764683821712,
            'stop_auto_buy': int(form.stop_auto_buy.data) if form.stop_auto_buy.data else 0.963808038695129,
            'rev_stop': int(form.rev_stop.data) if form.rev_stop.data else 0.0004330925354635,
            
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
        
        with torch.no_grad():
            output = model1(input_tensor)
        pytorch_pred = output.argmax(dim=1).item()

        if pytorch_pred == 1:
            num_trucks = form.num_trucks.data
            optimal_path, summary = calculate_optimal_path(num_trucks)
            return render_template("result.html", title="Result", backorder=True, path=optimal_path, summary=summary)
        else:
            return render_template("result.html", title="Result", backorder=False)
    return render_template("predict.html", title="Predict", form=form)

def calculate_optimal_path(num_trucks):
    # Mock data for companies
    optimal_path = num_trucks
    summary = "Optimal path calculated using given constraints."

    return optimal_path, summary

if __name__ == "__main__":
    app.run(debug=True)
