from flask import Flask, render_template, redirect, url_for, jsonify
from bs4 import BeautifulSoup
from tourney_simulator import season_simulator
from formatting import format
from chalk import chalk_simulator

app = Flask(__name__)

year = 2019
models = ["Model A", "Model B", "Model C"]
first64 = [str(x) for x in range(64)]
data = season_simulator(year)
model_A = format(data)
print("HERE")
model_B = chalk_simulator(year)

@app.route('/')
def render():
	return render_template("index.html", game = [])


@app.route('/modelA')
def modelA():
	return render_template("index.html", model="2019 Model", game = model_A)

@app.route('/modelB')
def modelB():
	return render_template("index.html", model="Chalk", game = model_B)

@app.route('/modelC')
def modelC():
	return render_template("index.html", model="Model C", game = first64)

if __name__ == '__main__':
	app.run()
