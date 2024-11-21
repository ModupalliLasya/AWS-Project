import json
import boto3
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Set up AWS Lambda and S3 client
lambda_client = boto3.client('lambda')
rekognition_client = boto3.client('rekognition')
polly_client = boto3.client('polly')
s3_client = boto3.client('s3')

# Your S3 bucket name
S3_BUCKET_NAME = "label-detection-bucket"

@app.route('/detect-labels', methods=['POST'])
def detect_labels():
    try:
        # Parse the form data
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save file temporarily
        file_name = secure_filename(file.filename)
        file_stream = file.read()

        # Upload file to S3 for Rekognition to access it
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=file_name, Body=file_stream)
        
        # Call AWS Rekognition to detect labels in the image
        response = rekognition_client.detect_labels(
            Image={'S3Object': {'Bucket': S3_BUCKET_NAME, 'Name': file_name}},
            MaxLabels=10,
            MinConfidence=50
        )

        labels = response['Labels']

        # Call Polly to generate an audio file for the labels
        labels_text = " ".join([label['Name'] for label in labels])
        polly_response = polly_client.synthesize_speech(
            Text=labels_text,
            OutputFormat="mp3",
            VoiceId="Joanna"
        )

        # Save the audio file to S3
        audio_file_name = "example.mp3"
        audio_url = upload_to_s3(audio_file_name, polly_response['AudioStream'])

        return jsonify({
            'labels': labels,
            'audio_url': audio_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def upload_to_s3(file_name, audio_stream):
    try:
        s3_client.upload_fileobj(audio_stream, S3_BUCKET_NAME, file_name)
        audio_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{file_name}"
        return audio_url
    except Exception as e:
        raise Exception(f"Failed to upload to S3: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
