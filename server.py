from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

app = Flask(__name__)

# Function to perform object detection
def perform_object_detection(image, endpoint_url):
    buffered = cv2.imencode('.jpg', image)[1].tobytes()

    # Build multipart form data for upload
    multipart_encoder = MultipartEncoder(
        fields={'file': ('image.jpg', buffered, 'image/jpeg')}
    )

    # Post the image to the detection endpoint
    response = requests.post(
        endpoint_url,
        data=multipart_encoder,
        headers={'Content-Type': multipart_encoder.content_type}
    )

    # Parse the response
    data = response.json()
    class_names = [prediction['class'] for prediction in data['predictions']]

    # Annotate the image with detected bounding boxes
    for prediction in data['predictions']:
        x = int(prediction['x'])
        y = int(prediction['y'])
        width = int(prediction['width'])
        height = int(prediction['height'])
        start_point = (x - width // 2, y - height // 2)
        end_point = (x + width // 2, y + height // 2)
        
        # Color based on object type
        color = (0, 0, 255) if prediction['class'] in ["accident", "knife", "Suspicious Behavior"] else (0, 255, 0)
        
        # Draw the rectangle and class name with confidence
        cv2.rectangle(image, start_point, end_point, color, 2)
        cv2.putText(
            image,
            f"{prediction['class']}: {prediction['confidence']:.2f}",
            (start_point[0], start_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    # Return the annotated image and class names
    return image, class_names


@app.route('/upload/weapon', methods=['POST'])
def upload_weapon():
    return handle_upload("https://detect.roboflow.com/weapon-detection-epu7t/1?api_key=arizb4SL99EVK1UYCG3i")


@app.route('/upload/car_crash', methods=['POST'])
def upload_car_crash():
    return handle_upload("https://detect.roboflow.com/car-crash-t32rg/1?api_key=arizb4SL99EVK1UYCG3i")


@app.route('/upload/shoplifting', methods=['POST'])
def upload_shoplifting():
    return handle_upload("https://detect.roboflow.com/shoplifting-sbnqg/1?api_key=arizb4SL99EVK1UYCG3i")

def handle_upload(endpoint_url):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        annotated_image, class_names = perform_object_detection(image, endpoint_url)

        # Convert the annotated image to base64
        _, img_encoded = cv2.imencode('.jpg', annotated_image)
        annotated_image_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({
            'annotated_image': annotated_image_base64,
            'class_names': class_names
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
