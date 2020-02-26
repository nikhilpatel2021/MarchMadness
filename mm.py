from flask import Flask, render_template, redirect, url_for, jsonify
from bs4 import BeautifulSoup

app = Flask(__name__)

models = ["Model A", "Model B", "Model C"]
currModel = 0

@app.route('/')
def render():
	return render_template("index.html")


@app.route('/modelA')
def modelA():
	return render_template("modelA.html")

@app.route('/modelB')
def modelB():
	return render_template("modelB.html")

@app.route('/modelC')
def modelC():
	return render_template("modelC.html")

if __name__ == '__main__':
	app.run()
