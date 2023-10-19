import face_recognition
import requests
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64

application = Flask(__name__)


@application.route('/compare_faces', methods=['POST'])
def compare_faces():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract the URL for the known image and Base64-encoded unknown image from the JSON data
        known_image_url = data['known_image_url']
        unknown_image_base64 = data['unknown_image_base64']

        # Download the known image from the URL
        response_known = requests.get(known_image_url)

        # Check if the download was successful
        if response_known.status_code == 200:
            # Convert the downloaded image content to a PIL Image
            known_image = Image.open(BytesIO(response_known.content))

            # Convert the Base64-encoded unknown image to bytes
            unknown_image_bytes = base64.b64decode(unknown_image_base64)

            # Convert the unknown image bytes to a PIL Image
            unknown_image = Image.open(BytesIO(unknown_image_bytes))

            # Convert PIL Images to numpy arrays for face_recognition
            known_image_np = face_recognition.api.load_image_file(BytesIO(response_known.content))
            unknown_image_np = face_recognition.api.load_image_file(BytesIO(unknown_image_bytes))

            known_encodings = face_recognition.face_encodings(known_image_np)

            if len(known_encodings) > 0:
                biden_encoding = known_encodings[0]
                unknown_encodings = face_recognition.face_encodings(unknown_image_np)

                if len(unknown_encodings) > 0:
                    unknown_encoding = unknown_encodings[0]
                    results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
                    result_str = str(results).strip('[]')

                    # Create a response dictionary with the specified format
                    response_data = {
                        'message': 'Comparison completed successfully.',
                        'status': result_str
                    }

                    return jsonify(response_data)
                else:
                    response_data = {
                        'status': 'error',
                        'message': 'No faces were found in the unknown image.'
                    }
                    return jsonify(response_data), 400
            else:
                response_data = {
                    'status': 'error',
                    'message': 'No faces were found in the known image.'
                }
                return jsonify(response_data), 400
        else:
            response_data = {
                'status': 'error',
                'message': 'Failed to download the known image from the provided URL.'
            }
            return jsonify(response_data), 400

    except Exception as e:
        response_data = {
            'status': 'error',
            'message': str(e)
        }
        return jsonify(response_data), 500


if __name__ == '__main__':
    application.run(debug=True,port=8080)
