from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum


# Common schemas
class StatusResponse(BaseModel):
    status: str
    version: str


class ModuleInfo(BaseModel):
    name: str
    description: str


class ModuleListResponse(BaseModel):
    modules: List[ModuleInfo]
    count: int


class ProviderType(str, Enum):
    openai = "openai"
    gemini = "gemini"
    huggingface = "huggingface"


# Supervised Learning schemas
class SupervisedTrainRequest(BaseModel):
    data: Union[List[Dict[str, Any]], str] = Field(..., description="Training data as list of dictionaries or CSV string")
    target_column: str = Field(..., description="Name of the target column")
    feature_columns: Optional[List[str]] = Field(None, description="Names of feature columns (if not provided, all non-target columns are used)")
    algorithm: str = Field(..., description="Algorithm to use (random_forest, logistic, svm, decision_tree, xgboost)")
    test_size: float = Field(0.2, description="Proportion of data to use for testing")
    random_state: Optional[int] = Field(None, description="Random seed for reproducibility")
    params: Optional[Dict[str, Any]] = Field(None, description="Algorithm-specific parameters")


class TrainResponse(BaseModel):
    model_id: str
    algorithm: str
    metrics: Dict[str, Any]
    elapsed_time: float


class PredictRequest(BaseModel):
    model_id: str = Field(..., description="ID of the trained model")
    data: Union[List[Dict[str, Any]], str] = Field(..., description="Data to make predictions on")


class PredictResponse(BaseModel):
    predictions: List[Any]
    model_id: str
    confidence: Optional[List[float]] = None


# Unsupervised Learning schemas
class ClusterRequest(BaseModel):
    data: Union[List[Dict[str, Any]], str] = Field(..., description="Data to cluster")
    algorithm: str = Field(..., description="Clustering algorithm (kmeans, dbscan, hierarchical)")
    n_clusters: Optional[int] = Field(None, description="Number of clusters for kmeans and hierarchical")
    params: Optional[Dict[str, Any]] = Field(None, description="Algorithm-specific parameters")


class ClusterResponse(BaseModel):
    cluster_labels: List[int]
    algorithm: str
    metrics: Optional[Dict[str, Any]] = None


# NLP schemas
class TextInputRequest(BaseModel):
    text: str = Field(..., description="Input text")
    provider: Optional[ProviderType] = Field(ProviderType.huggingface, description="API provider")
    model: Optional[str] = Field(None, description="Model ID or name")
    params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")


class TextOutputResponse(BaseModel):
    text: str
    provider: ProviderType
    model: Optional[str] = None


class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_language: Optional[str] = Field(None, description="Source language code")
    target_language: str = Field(..., description="Target language code")
    provider: Optional[ProviderType] = Field(ProviderType.huggingface, description="API provider")
    model: Optional[str] = Field(None, description="Model ID or name")


# Computer Vision schemas
class ImageInputRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="URL to the image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    provider: Optional[ProviderType] = Field(ProviderType.huggingface, description="API provider")
    model: Optional[str] = Field(None, description="Model ID or name")
    
    @root_validator
    def check_image_source(cls, values):
        """Validate that either image_url or image_base64 is provided"""
        if not values.get('image_url') and not values.get('image_base64'):
            raise ValueError("Either image_url or image_base64 must be provided")
        return values


class CVClassifyResponse(BaseModel):
    labels: List[str]
    scores: List[float]
    provider: ProviderType
    model: Optional[str] = None


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectionResult(BaseModel):
    label: str
    score: float
    box: BoundingBox


class CVDetectResponse(BaseModel):
    detections: List[DetectionResult]
    provider: ProviderType
    model: Optional[str] = None


# Time Series schemas
class TimeSeriesRequest(BaseModel):
    data: Union[List[Dict[str, Any]], str] = Field(..., description="Time series data")
    timestamp_column: str = Field(..., description="Name of the timestamp column")
    value_column: str = Field(..., description="Name of the value column")
    algorithm: str = Field(..., description="Forecasting algorithm (arima, sarima, prophet, exponential, lstm)")
    steps: int = Field(..., description="Number of steps to forecast")
    params: Optional[Dict[str, Any]] = Field(None, description="Algorithm-specific parameters")


class ForecastResponse(BaseModel):
    timestamps: List[str]
    forecasted_values: List[float]
    algorithm: str
    confidence_intervals: Optional[Dict[str, List[float]]] = None


# Reinforcement Learning schemas
class RLTrainRequest(BaseModel):
    env_name: str = Field(..., description="Name of the Gym environment")
    algorithm: str = Field(..., description="RL algorithm (q_learning, dqn, policy_gradient)")
    episodes: int = Field(500, description="Number of episodes for training")
    params: Optional[Dict[str, Any]] = Field(None, description="Algorithm-specific parameters")


# LLM Tools schemas
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    provider: Optional[ProviderType] = Field(ProviderType.openai, description="API provider")
    model: Optional[str] = Field(None, description="Model ID or name")
    temperature: Optional[float] = Field(0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for response")


class ChatResponse(BaseModel):
    response: str
    provider: ProviderType
    model: str


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    provider: Optional[ProviderType] = Field(ProviderType.openai, description="API provider")
    model: Optional[str] = Field(None, description="Model ID or name")
    temperature: Optional[float] = Field(0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for response")
    params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")


# Sensor & Signal schemas
class SensorDataRequest(BaseModel):
    data: Union[List[Dict[str, Any]], str] = Field(..., description="Sensor data")
    timestamp_column: str = Field(..., description="Name of the timestamp column")
    feature_columns: List[str] = Field(..., description="Names of feature columns")
    algorithm: str = Field(..., description="Anomaly detection algorithm (isolation_forest, lof, dbscan)")
    params: Optional[Dict[str, Any]] = Field(None, description="Algorithm-specific parameters")


class AnomalyResponse(BaseModel):
    anomaly_indices: List[int]
    anomaly_scores: List[float]
    algorithm: str


# TinyML schemas
class TFLitePathRequest(BaseModel):
    model_path: str = Field(..., description="Path to the TFLite model")
    output_path: Optional[str] = Field(None, description="Path to save the C array header file")


class FileDownloadResponse(BaseModel):
    file_path: str
    file_size: int
    content_type: str 