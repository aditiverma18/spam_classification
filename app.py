from flask import Flask, render_template, request, jsonify
from model import predict_spam  # Import your classification function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        message = request.form.get('message')
        if message:
            # You may need to adjust how you call your model
            result = predict_spam(message)
    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    message = data.get('message', '')
    prediction = predict_spam(message)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
