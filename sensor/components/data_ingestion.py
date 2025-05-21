from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataIngestionArtifact
from sensor.entity.config_entity import DataIngestionConfig
from sklearn.model_selection import train_test_split
import os, sys
import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv

# Load environment variables (especially for MongoDB connection)
load_dotenv()

# Fetch MongoDB URL from environment
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Reads a MongoDB collection and converts it to a pandas DataFrame.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop("_id", axis=1)

            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Exporting collection as dataframe")
            df = self.export_collection_as_dataframe()

            logging.info("Splitting dataframe into train and test")
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size)

            logging.info("Creating dataset directory")
            os.makedirs(self.data_ingestion_config.dataset_dir, exist_ok=True)

            logging.info("Saving train and test file.")
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info("Preparing data ingestion artifact")
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)
