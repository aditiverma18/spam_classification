from flask import Flask, render_template_string, request, jsonify

# Dummy spam classifier function; replace with your actual logic/model
def predict_spam(message):
    if message and 'free' in message.lower():
        return 'Spam'
    return 'Not Spam'

app = Flask(__name__)

# Read HTML and CSS from files
with open("index.html", "r") as f:
    html_content = f.read()
with open("styl.css", "r") as f:
    css_content = f.read()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        message = request.form.get("message")
        if message:
            result = predict_spam(message)
    # Inject CSS into HTML for simplicity
    page = html_content.replace("<!--STYLE_HERE-->", f"<style>{css_content}</style>")
    return render_template_string(page, result=result)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    message = data.get('message', '')
    prediction = predict_spam(message)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
