# # from flask import Flask, request, jsonify
# # from tensorflow.keras.models import load_model
# # import cv2
# # import numpy as np

# # app = Flask(__name__)

# # # Load your model
# # model = load_model('ultra.keras')

# # # Define classes (e.g., Mountain, Glacier, etc.)
# # classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     if 'image' not in request.files:
# #         return jsonify({'error': 'No image uploaded'}), 400

# #     # Read the image
# #     file = request.files['image']
# #     image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
# #     image_resized = cv2.resize(image, (224, 224))  # Match model input size
# #     image_normalized = image_resized / 255.0
# #     image_reshaped = np.expand_dims(image_normalized, axis=0)

# #     # Make prediction
# #     predictions = model.predict(image_reshaped)
# #     class_index = np.argmax(predictions)
# #     predicted_class = classes[class_index]

# #     return jsonify({'class': predicted_class})

# # if __name__ == '__main__':
# #     app.run(debug=True)






# from flask import Flask, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# app = Flask(__name__)

# # Load the pre-trained model
# model = load_model('ultra.keras')
# height = 150
# width = 150

# # Class labels
# class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# # Function to preprocess the image
# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(height, width))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  
#     img_array = img_array / 255.0  
#     return img_array

# # Function to predict the image class
# def predict_image_class(img_path):
#     img_array = preprocess_image(img_path)
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions, axis=1)
#     return predicted_class[0]

# # API endpoint for image prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400

#     file = request.files['image']

#     if file.filename == '':
#         return jsonify({'error': 'No file selected for uploading'}), 400

#     try:
#         # Save the uploaded file temporarily
#         file_path = os.path.join("uploads", file.filename)
#         os.makedirs("uploads", exist_ok=True)
#         file.save(file_path)

#         # Predict the class of the image
#         predicted_class = predict_image_class(file_path)
#         result = {
#             'predicted_class': class_names[predicted_class],
#             'confidence_scores': model.predict(preprocess_image(file_path)).tolist()
#         }

#         # Remove the temporary file
#         os.remove(file_path)

#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Run the Flask app
# if __name__ == "__main__":
#     app.run(port=5000, debug=True)



from flask import Flask, request, jsonify
from flask_cors import CORS  # For Cross-Origin requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the pre-trained model
model = load_model('ultra.keras')
height, width = 150, 150

# Class labels
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(height, width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

# Function to predict the image class
def predict_image_class(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0], predictions[0].tolist()

# API endpoint for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Save the uploaded file temporarily
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        # Predict the class of the image
        predicted_class, confidence_scores = predict_image_class(file_path)
        result = {
            'predicted_class': class_names[predicted_class],
            'confidence_scores': confidence_scores
        }

        # Remove the temporary file
        os.remove(file_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000, debug=True)