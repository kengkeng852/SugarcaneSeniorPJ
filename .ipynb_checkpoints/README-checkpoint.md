# Sugarcane Disease Detection API

This repository contains a FastAPI application for sugarcane disease detection using a TensorFlow model. The application allows users to upload images of sugarcane leaves and receive predictions about the type of disease affecting the leaves.

## Project Overview

The purpose of this project is to provide an API for detecting diseases in sugarcane leaves. The application uses a pre-trained TensorFlow model to classify images of sugarcane leaves into different disease categories.

## Setup Instructions

### Prerequisites

- Python 3.10.11
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kengkeng852/SugarcaneSeniorPJ.git
   cd SugarcaneSeniorPJ/api
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

### Running the Application Locally

1. Start the FastAPI application:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. Open your web browser and navigate to `http://localhost:8000` to access the API.

### API Endpoints

- `GET /`: Returns a welcome message.
- `POST /predict`: Accepts an image file and returns the predicted disease label and confidence score.
- `POST /webhook`: Handles webhook events from LINE messaging API.

## Deployment Instructions

### Deploying to Google Cloud Run

1. Build and deploy the application using the following command:

   ```bash
   gcloud run deploy sugarcane-predict-api --source "C:\Users\kongp\Downloads\SugarcaneSeniorPJ\api" --region asia-southeast1 --allow-unauthenticated --platform managed
   ```

2. Follow the instructions provided by Google Cloud Run to complete the deployment.

3. Once deployed, you can access the API using the URL provided by Google Cloud Run.

## Additional Information

- The TensorFlow model used for predictions is located at `api/best_model8.keras`.
- The `api/Dockerfile` sets up the environment for the application.
- Dependencies are listed in `api/requirements.txt`.
- The `api/buildpack.yml` and `api/runtime.txt` specify the Python runtime version.

