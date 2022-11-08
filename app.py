from flask import Flask, render_template, request
from inference import decode_sequence
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        r = request.form['url']

    else:
        r = None
    if r:
        r = decode_sequence(r)
        r = r.replace(r"[start]", "")
        r = r.replace(r"[end]", "")
        #print(r.replace(r"[end]", ""))
    print(r)
    return render_template('index.html', processed_text=r)


def my_form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text
