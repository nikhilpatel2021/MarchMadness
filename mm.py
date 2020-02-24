from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "<h1>March Madness 2020</h1><p>Nikhil Patel - Radin Marinov</p>"
