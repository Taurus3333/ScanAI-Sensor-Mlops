import os
import sys
import certifi
from dotenv import load_dotenv
import uvicorn 

from fastapi import FastAPI, Response, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
import pandas as pd

from sensor.logger import logging
from sensor.pipeline.training_pipeline import *
from sensor.ml.estimator import  TargetValueMapping
from sensor.ml.model_resolver import ModelResolver
from sensor.utils import load_object
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
from sensor.constant.application import APP_HOST, APP_PORT

import pymongo

# Load environment variables from .env file
load_dotenv()

# MongoDB connection string from .env
mongo_db_url = os.getenv("MONGO_DB_URL")
if not mongo_db_url:
    raise Exception("MONGO_DB_URL not set in environment variables")

# Use certifi to get CA bundle path for TLS/SSL connection
ca = certifi.where()

# Create MongoDB client with TLS CA file for secure connection
mongo_client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

# Example: Access database and collection if needed
# from sensor.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_COLLECTION_NAME
# database = mongo_client[DATA_INGESTION_DATABASE_NAME]
# collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
def train():
    try:
        training_pipeline_config = TrainingPipelineConfig()
        pipeline = TrainingPipeline(training_pipleine_config=training_pipeline_config)
        pipeline.start()
        return {"message": "Training successful!"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Read uploaded CSV file as DataFrame
        df = pd.read_csv(file.file)

        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available")

        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)

        y_pred = model.predict(df)
        df['predicted_column'] = y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(), inplace=True)


    except Exception as e:
        logging.exception(e)
        return Response(f"Error Occurred! {e}")

if __name__ == "__main__":
    app_run(app, host="127.0.0.1", port=8080)

