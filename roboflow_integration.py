import requests
import cv2
import os
import tempfile

class RoboflowPredictor:
    def __init__(self, api_key, model_id, version_number):
        self.api_key = api_key
        self.model_id = model_id
        self.version_number = version_number
        self.api_url = f"https://detect.roboflow.com/{model_id}/{version_number}"
        self.params = {"api_key": api_key}
        
    def predict_frame(self, frame):
        """Predict sign language from a frame"""
        try:
            # Save frame to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, frame)
            
            # Make API request
            with open(temp_path, "rb") as image_file:
                response = requests.post(
                    self.api_url,
                    params=self.params,
                    files={"file": image_file}  # ✅ Correct way
    
                    )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            if response.status_code == 200:
                return self.parse_prediction(response.json())
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None, 0
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0
    
    def parse_prediction(self, prediction_data):
        """Extract the predicted sign from the response"""
        if not prediction_data or "predictions" not in prediction_data:
            return None, 0
        
        predictions = prediction_data["predictions"]
        if not predictions:
            return None, 0
        
        # Get the prediction with highest confidence
        best_prediction = max(predictions, key=lambda x: x["confidence"])
        return best_prediction["class"], best_prediction["confidence"]