from flask import Flask

app = Flask(__name__)
from flask import Flask, render_template


@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/')
def index():
    return render_template('index.html')