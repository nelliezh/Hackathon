from flask import Flask
app = Flask(__name__)
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
from flask import Flask, render_template, request
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    if request.method == "POST":
        # Identify which button was clicked
        if "btn_hello" in request.form:
            message = "Hello button clicked!"
        elif "btn_goodbye" in request.form:
            message = "Goodbye button clicked!"
        elif "btn_custom" in request.form:
            message = "Custom action executed!"
    return render_template("index.html", message=message)
@app.route("/picture")
def capture():
    pass
if __name__ == "__main__":
    app.run(debug=True)