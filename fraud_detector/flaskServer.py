from flask import Flask
import fraudDector as fd

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/dtree/<int:requestID>')
def get_dtree(requestID):
    request_time = requestID
    model = model_updater.get_model(request_time)
    return fd.get_dtree_parameter(model)


@app.route('/scale/<int:requestID>')
def get_scaler(requestID):
    request_time = requestID
    scaler = model_updater.get_scaler(request_time)
    return fd.get_scale_rule(scaler)


import pandas as pd

df = pd.read_csv('arrange.csv')
print(df.columns)
df.drop('Unnamed: 0', axis=1, inplace=True)

model_updater = fd.ModelUpdater(df)
model_updater.start()
