import os
from flask import Flask, render_template, request, redirect, url_for
from model_loader import predict

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_image = None

@app.route("/", methods=["GET", "POST"])
def index():
    global current_image
    answer = None

    if request.method == "POST":
        file = request.files.get("image")
        question = request.form.get("question")

        # Save uploaded image if present
        if file and file.filename != "":
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            current_image = filepath

        # Run prediction if image and question exist
        if current_image and question:
            answer = predict(current_image, question)

    # Show image if uploaded, regardless of answer
    return render_template("index.html", answer=answer, image_path=current_image)

@app.route("/reset", methods=["GET", "POST"])
def reset():
    global current_image
    current_image = None
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)