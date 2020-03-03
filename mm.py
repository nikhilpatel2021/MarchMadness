from flask import Flask, render_template, redirect, url_for, jsonify
from bs4 import BeautifulSoup
from tourney_simulator import season_simulator
from formatting import format

app = Flask(__name__)

models = ["Model A", "Model B", "Model C"]
first64 = [str(x) for x in range(64)]
currModel = 0
data = season_simulator(2019)
formatted_data = format(data)
print(formatted_data)

@app.route('/')
def render():
	return render_template("index.html", game = first64)


@app.route('/modelA')
def modelA():
	return render_template("index.html", model="Model A", game = formatted_data)

@app.route('/modelB')
def modelB():
	return render_template("index.html", model="Model B", game = first64)

@app.route('/modelC')
def modelC():
	return render_template("index.html", model="Model C", game = first64)

if __name__ == '__main__':
	app.run()
