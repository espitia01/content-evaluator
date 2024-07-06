from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
from preprocess_and_predict import preprocess_and_predict
import warnings
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app)

warnings.filterwarnings("ignore", category=UserWarning)

try:
    model_data = joblib.load('youtube_predictor_model_2.joblib')
    print("Model loaded successfully")
    print("Structure of model_data:", {k: type(v) for k, v in model_data.items()})
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_data = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        form_data = request.json
        print("Received prediction request with data:", form_data)
        
        # Construct the video_data object
        video_data = {
            "channel_subscriber_count": form_data['channel_subscriber_count'],
            "channel_video_count": form_data['channel_video_count'],
            "channel_view_count": form_data['channel_view_count'],
            "channel_created_at": form_data['channel_created_at'],
            "duration": f"PT{form_data['duration']}S",
            "collection_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        results = preprocess_and_predict(video_data, model_data['scaler_X'], model_data['selected_features'], model_data['final_models'], model_data['pt'])
        print("Sending prediction results:", results)
        return jsonify({k: round(v, 2) for k, v in results.items()})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=3000)