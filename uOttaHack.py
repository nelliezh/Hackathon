from flask import Flask, send_file
import subprocess
import time
import io

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
    # Start the make_zero.sh script
    process = subprocess.Popen(
        ["./make_zero.sh"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(0.5)

    process.stdin.write(b"B")
    process.stdin.flush()

    time.sleep(0.5)
    process.kill()

    with open("pi.png", "rb") as f:
        image_data = f.read()

    print("Returning picture")
    return send_file(io.BytesIO(image_data), mimetype="image/jpeg", as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True)
