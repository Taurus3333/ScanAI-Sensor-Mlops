from sensor.entity.config_entity import (TrainingPipelineConfig,
                                        DataIngestionConfig,
                                        DataValidationConfig,
                                        DataTransformationConfig,
                                        ModelTrainerConfig,
                                        ModelEvaluationConfig,
                                        ModelPusherConfig
                                        )
from sensor.entity.artifact_entity import (DataIngestionArtifact,
                                        DataValidationArtifact,
                                        DataTransformationArtifact,
                                        ModelTrainerArtifact,
                                        ModelEvaluationArtifact,
                                        ModelPusherArtifact
                                        )
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.cloud.s3_syncer import S3Sync
# from sensor.constant.training_pipeline import TRAINING_BUCKET_NAME
from sensor.constant.training_pipeline import SAVED_MODEL_DIR

TRAINING_BUCKET_NAME = "sensor27042025"



class TrainingPipeline:

    is_pipeline_running = False

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        try:
            self.training_pipeline_config=training_pipeline_config
            self.s3_sync = S3Sync()
        except Exception as e:
            raise SensorException(e, sys)

    # class TrainingPipeline:

    #     is_pipeline_running = False

    #     def __init__(self, training_pipeline_config: TrainingPipelineConfig):
    #         try:
    #             self.training_pipeline_config = training_pipeline_config
    #             self.s3_sync = S3Sync()
    #         except Exception as e:
    #             raise SensorException(e, sys)



    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion_config =DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config)
            
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config=data_validation_config,
             data_ingestion_artifact=data_ingestion_artifact)
            
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise SensorException(e, sys)

    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
             data_validation_artifact=data_validation_artifact)
            
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise SensorException(e, sys)

    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
             data_transformation_artifact=data_transformation_artifact)
            
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise SensorException(e, sys)

    def start_model_evaluation(self,   data_validation_artifact: DataValidationArtifact,
        data_transformation_artifact:DataTransformationArtifact,
        model_trainer_artifact:ModelTrainerArtifact)->ModelEvaluationArtifact:
        try:
            model_eval_config = ModelEvaluationConfig(training_pipeline_config=self.training_pipeline_config)
            model_eval = ModelEvaluation(model_eval_config=model_eval_config,
             data_validation_artifact=data_validation_artifact,
             data_transformation_artifact=data_transformation_artifact,
             model_trainer_artifact=model_trainer_artifact)
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise SensorException(e, sys)

    def start_model_pusher(self,  data_transformation_artifact:DataTransformationArtifact,
        model_trainer_artifact:ModelTrainerArtifact)->ModelPusherArtifact:
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
             data_transformation_artifact=data_transformation_artifact, 
             model_trainer_artifact=model_trainer_artifact)
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise SensorException(e, sys)

    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e,sys)
            
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(folder = SAVED_MODEL_DIR,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e,sys)


    def start(self):

        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact)

            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )

            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            

            model_eval_artifact = self.start_model_evaluation(data_validation_artifact=data_validation_artifact,
                            data_transformation_artifact=data_transformation_artifact,
                            model_trainer_artifact=model_trainer_artifact)

            model_pusher_artifact = self.start_model_pusher(data_transformation_artifact=data_transformation_artifact,
                            model_trainer_artifact=model_trainer_artifact)
            
            self.sync_artifact_dir_to_s3()

            self.sync_saved_model_dir_to_s3()

        except Exception as e:
            raise SensorException(e, sys)





    
        