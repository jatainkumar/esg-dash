import networkx as nx
import geopandas as gpd
import folium
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import io
import base64
from flask import render_template
import time
import random

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import CplexOptimizer
from IPython.display import display
from docplex.mp.model import Model
import geodatasets
from geodatasets import get_path


import torch
import torch.nn as nn
import torch.nn.functional as F



from flask import Flask, url_for, render_template, request
from forms import InputForm1, InputForm2

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

@app.route("/predict1", methods=["GET", "POST"])
def predict1():
    form = InputForm1()
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

        input_df = pd.DataFrame([input_data])
        input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
        
        with torch.no_grad():
            output = model1(input_tensor)
        pytorch_pred = output.argmax(dim=1).item()

        if pytorch_pred == 0:
            return render_template("predict2.html", title="Predict2", form=InputForm2())
        else:
            return render_template("result.html", title="Result", backorder=False)
    return render_template("predict1.html", title="Predict1", form=form)

@app.route("/predict2", methods=["GET", "POST"])
def predict2():
    form = InputForm2()
    num_trucks = int(request.form['num_trucks'])
    
    #LOCATIONS ON MAP
    path_to_file = get_path('nybb')
    df = gpd.read_file(path_to_file)
    m = folium.Map(location=[40.70, -73.94], zoom_start=10, max_zoom=12, tiles='CartoDB positron')
    df = df.to_crs(epsg=2263)
    df['centroid'] = df.centroid
    df = df.to_crs(epsg=4326)
    df['centroid'] = df['centroid'].to_crs(epsg=4326)
    np.random.seed(7)
    locations = []
    ids = 0
    for _, r in df.iterrows():
        lat, lon = r['centroid'].y, r['centroid'].x
        for i in range(2):
            lat_rand, lon_rand = lat + 0.2 * np.random.rand(), lon +0.1 * np.random.rand()
            locations.append((lon_rand, lat_rand))
            folium.Marker(location=[lat_rand, lon_rand], popup=f'Id: {ids}').add_to(m)
            ids += 1
    center = np.array(locations).mean(axis=0)
    locations = [(center[0], center[1])] + locations
    folium.CircleMarker(location=[center[1], center[0]], radius=10, popup="<strong>Warehouse</strong>",
                        color="red", fill=True, fillOpacity=1, fillColor="tab:red").add_to(m)
    map_html = m._repr_html_()

    #NUMBERS OF EDGES
    companies = np.array(locations)
    companies -= companies[0]
    companies /= (np.max(np.abs(companies), axis=0))
    r = list(np.sqrt(np.sum(companies ** 2, axis=1)))
    threshold = 1
    n_companies = len(companies)
    G = nx.Graph(name="VRP")
    G.add_nodes_from(range(n_companies))
    np.random.seed(2)
    count = 0
    for i in range(n_companies):
        for j in range(n_companies):
            if i != j:
                rij = np.sqrt(np.sum((companies[i] - companies[j])**2))
                if (rij < threshold) or (0 in [i, j]):
                    count +=1
                    G.add_weighted_edges_from([[i, j, rij]])
                    r.append(rij)
    colors = [plt.cm.get_cmap("coolwarm")(x) for x in r[1:]]
    nx.draw(G, pos=companies, with_labels=True, node_size=500,
            edge_color=colors, width=1, font_color="white",font_size=14,
            node_color = ["tab:red"] + (n_companies-1)*["darkblue"])
    edges=len(G.edges)   

    #GRAPH1
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    graph1 = base64.b64encode(img.getvalue()).decode('utf8')

    #SOLVE THE PROBLEM
    mdl = Model(name="VRP")
    n_trucks= num_trucks
    x = {}
    for i, j in G.edges():
        x[(i, j)] = mdl.binary_var(name=f"x_{i}_{j}")
        x[(j, i)] = mdl.binary_var(name=f"x_{j}_{i}")

    print(f"The number of qubits needed to solve the problem is: {mdl.number_of_binary_variables}")
    cost_func = mdl.sum(w["weight"] * x[(i, j)] for i, j, w in G.edges(data=True)) + \
                mdl.sum(w["weight"] * x[(j, i)] for i, j, w in G.edges(data=True))
    mdl.minimize(cost_func)

    # Constraint 1a(yellow Fig. above): Only one truck goes out from company i 
    for i in range(1, n_companies):
        mdl.add_constraint(mdl.sum(x[i, j] for j in range(n_companies) if (i, j) in x.keys()) == 1)
    
    # Constraint 1b (yellow Fig. above): Only one truck comes into company j 
    for j in range(1, n_companies):
        mdl.add_constraint(mdl.sum(x[i, j] for i in range(n_companies) if (i, j) in x.keys()) == 1)
    
    # Constraint 2: (orange Fig. above) For the warehouse
    mdl.add_constraint(mdl.sum(x[i, 0] for i in range(1, n_companies)) == num_trucks)
    mdl.add_constraint(mdl.sum(x[0, j] for j in range(1, n_companies)) == num_trucks)

    # Constraint 3: (blue Fig. above) To eliminate sub-routes
    companies_list = list(range(1, n_companies))
    subroute_set = []
    for i in range(2, len(companies_list) + 1):
        for comb in itertools.combinations(companies_list, i):
            subroute_set.append(list(comb))
    
    for subroute in subroute_set:
        constraint_3 = []
        for i, j in itertools.permutations(subroute, 2):
            if (i, j) in x.keys():
                constraint_3.append(x[(i,j)])
            elif i == j:
                pass
            else:
                constraint_3 = []
                break
        if len(constraint_3) != 0:
            mdl.add_constraint(mdl.sum(constraint_3) <= len(subroute) - 1)

    quadratic_program = from_docplex_mp(mdl)
    cost_functions=quadratic_program.export_as_lp_string()


    #GRAPH2
    sol = CplexOptimizer().solve(quadratic_program)
    solution_cplex = sol.raw_results.as_name_dict()
    G_sol = nx.Graph()
    G_sol.add_nodes_from(range(n_companies))
    for i in solution_cplex:
        nodes = i[2:].split("_")
        G_sol.add_edge(int(nodes[0]), int(nodes[1]))
    nx.draw(G_sol, pos=companies, with_labels=True, node_size=500,
            edge_color=colors, width=1, font_color="white",font_size=14,
            node_color = ["tab:red"] + (n_companies-1)*["darkblue"])
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    plt.close()
    img2.seek(0)
    graph2 = base64.b64encode(img2.getvalue()).decode('utf8')




    #sol = CplexOptimizer().solve(quadratic_program)
    # solution_cplex = sol.raw_results.as_name_dict()
    # G_sol = nx.Graph()
    # G_sol.add_nodes_from(range(n_companies))
    # for i in solution_cplex:
    #     nodes = i[2:].split("_")
    #     G_sol.add_edge(int(nodes[0]), int(nodes[1]))

    # Prepare data to pass to the template
    #cost_functions = [f"x_{i}_{j}: {v}" for (i, j), v in x.items()]
    cost_summary = f"The number of qubits needed to solve the problem is: {mdl.number_of_binary_variables}"
    

    return render_template('result2.html', cost_functions=cost_functions, cost_summary=cost_summary, map_html=map_html, graph1=graph1, graph2=graph2, edges=edges)

if __name__ == "__main__":
    app.run(debug=True)
