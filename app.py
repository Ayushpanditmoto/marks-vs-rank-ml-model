from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your pre-trained Random Forest Regression model
model = joblib.load('best_model.pkl')
print('Model loaded')
@app.route('/predict_rank', methods=['POST'])
def predict_rank():
    try:
        # Get marks from the request
        marks = float(request.json['marks'])

        # Make a prediction using the loaded model
        rank_prediction = model.predict([[marks]])[0]

        # Return the predicted rank as JSON response
        response = {'predicted_rank': rank_prediction}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
