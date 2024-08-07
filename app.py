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
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
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
    
input_dim = 21  # Change this to match the input features of your model
model1 = Net(input_dim, 2)
model1.load_state_dict(torch.load('net_model_state_dict.pth'))
model1.eval()


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        input_data = pd.DataFrame({
            'national_inv': [form.national_inv.data],
            'lead_time': [form.lead_time.data],
            'forecast_3_month': [form.forecast_3_month.data]
        })
        input_tensor = torch.tensor(input_data.values, dtype=torch.float32)
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
    optimal_path= num_trucks
    summary= "Optimal path calculated using given constraints."

    return optimal_path, summary


if __name__ == "__main__":
    app.run(debug=True)
