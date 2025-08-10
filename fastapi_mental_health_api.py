# ================================
# FastAPI Mental Health Support API
# ================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import json
import joblib
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import numpy as np
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# Pydantic Models for API
# ================================

class TextInput(BaseModel):
    text: str
    max_predictions: Optional[int] = 3

class EmotionPrediction(BaseModel):
    emotion: str
    confidence: float

class EmotionResponse(BaseModel):
    emotions: List[EmotionPrediction]
    primary_emotion: str
    confidence: float
    timestamp: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str

class CombinedAnalysisResponse(BaseModel):
    text: str
    emotions: List[EmotionPrediction]
    primary_emotion: str
    sentiment: str
    sentiment_confidence: float
    analysis_timestamp: str

class HealthResponse(BaseModel):
    status: str
    emotion_model_loaded: bool
    sentiment_model_loaded: bool
    timestamp: str

# ================================
# Model Configuration
# ================================

class ModelConfig:
    def __init__(self):
        self.config = {
            "bert_emotion_model_path": "./Final/Final/best_model",
            "bert_sentiment_model_path": "./Final/Final/best_sentiment_model",
            "emotion_label_encoder_path": "./Final/Final/label_encoder.pkl",
            "device": "cpu"  # Force CPU for API deployment
        }
        self.sentiment_id_to_label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.emotion_model_loaded = False
        self.sentiment_model_loaded = False
        self.models_loaded = False
        
        # Initialize model variables
        self.emotion_tokenizer = None
        self.emotion_model = None
        self.emotion_label_encoder = None
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        
        self.load_models()

    def load_models(self):
        """Load models with memory optimization - try to load only what fits in memory"""
        try:
            logger.info("Loading models with memory optimization...")
            
            # Clear any existing GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Convert relative paths to absolute paths
            import os
            emotion_model_path = os.path.abspath(self.config["bert_emotion_model_path"])
            sentiment_model_path = os.path.abspath(self.config["bert_sentiment_model_path"])
            label_encoder_path = os.path.abspath(self.config["emotion_label_encoder_path"])
            
            # Verify paths exist
            if not os.path.exists(emotion_model_path):
                logger.warning(f"Emotion model path does not exist: {emotion_model_path}")
            if not os.path.exists(sentiment_model_path):
                logger.warning(f"Sentiment model path does not exist: {sentiment_model_path}")
            if not os.path.exists(label_encoder_path):
                logger.warning(f"Label encoder path does not exist: {label_encoder_path}")
            
            # Try to load emotion model first (usually more memory intensive)
            try:
                if os.path.exists(emotion_model_path):
                    logger.info(f"Loading emotion model from: {emotion_model_path}")
                    self.emotion_tokenizer = AutoTokenizer.from_pretrained(
                        emotion_model_path,
                        local_files_only=True
                    )
                    self.emotion_model = AutoModelForSequenceClassification.from_pretrained(
                        emotion_model_path,
                        local_files_only=True,
                        torch_dtype=torch.float32
                    ).to(self.config["device"]).eval()

                    # Load emotion label encoder
                    if os.path.exists(label_encoder_path):
                        self.emotion_label_encoder = joblib.load(label_encoder_path)
                        self.emotion_model_loaded = True
                        logger.info("Emotion model loaded successfully")
                    else:
                        logger.error(f"Label encoder not found at: {label_encoder_path}")
                        self.emotion_model_loaded = False
                else:
                    logger.error(f"Emotion model directory not found: {emotion_model_path}")
                    self.emotion_model_loaded = False
                
                # Force garbage collection after emotion model
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to load emotion model: {e}")
                self.emotion_model_loaded = False

            # Try to load sentiment model
            try:
                if os.path.exists(sentiment_model_path):
                    logger.info(f"Loading sentiment model from: {sentiment_model_path}")
                    self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                        sentiment_model_path,
                        local_files_only=True
                    )
                    self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                        sentiment_model_path,
                        local_files_only=True,
                        torch_dtype=torch.float32
                    ).to(self.config["device"]).eval()
                    self.sentiment_model_loaded = True
                    logger.info("Sentiment model loaded successfully")
                else:
                    logger.error(f"Sentiment model directory not found: {sentiment_model_path}")
                    self.sentiment_model_loaded = False
                
            except Exception as e:
                logger.error(f"Failed to load sentiment model: {e}")
                self.sentiment_model_loaded = False

            # Set overall status
            self.models_loaded = self.emotion_model_loaded or self.sentiment_model_loaded
            
            if self.emotion_model_loaded and self.sentiment_model_loaded:
                logger.info("All models loaded successfully")
            elif self.emotion_model_loaded:
                logger.warning("Only emotion model loaded - sentiment analysis will use fallback")
            elif self.sentiment_model_loaded:
                logger.warning("Only sentiment model loaded - emotion analysis will use fallback")
            else:
                logger.error("No models could be loaded - using fallback responses")
                
            # Final garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error in model loading process: {e}")
            self.emotion_model_loaded = False
            self.sentiment_model_loaded = False
            self.models_loaded = False

    def predict_emotion_top_n(self, text: str, top_n: int = 3):
        """Predict top N emotions with their probabilities"""
        if not self.emotion_model_loaded or self.emotion_model is None:
            return ["Content"], [1.0]
            
        try:
            inputs = self.emotion_tokenizer(
                text, 
                truncation=True, 
                padding='max_length', 
                max_length=128, 
                return_tensors='pt'
            )
            inputs = {k: v.to(self.config["device"]) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.emotion_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get top N predictions
                top_probs, top_indices = torch.topk(probs, top_n, dim=-1)
                top_probs = top_probs.squeeze().cpu().numpy()
                top_indices = top_indices.squeeze().cpu().numpy()
                
                # Convert indices to emotion labels
                top_emotions = []
                top_scores = []
                
                # Handle single prediction case
                if top_n == 1:
                    top_indices = [top_indices]
                    top_probs = [top_probs]
                
                for idx, prob in zip(top_indices, top_probs):
                    emotion_label = self.emotion_label_encoder.inverse_transform([idx])[0]
                    top_emotions.append(emotion_label)
                    top_scores.append(float(prob))
                
                return top_emotions, top_scores
                
        except Exception as e:
            logger.error(f"Error predicting emotion: {e}")
            return ["Content"], [1.0]

    def predict_sentiment(self, text: str):
        """Predict sentiment with confidence scores"""
        if not self.sentiment_model_loaded or self.sentiment_model is None:
            return "Neutral", 1.0, {"Negative": 0.0, "Neutral": 1.0, "Positive": 0.0}
            
        try:
            inputs = self.sentiment_tokenizer(
                text, 
                truncation=True, 
                padding='max_length', 
                max_length=128, 
                return_tensors='pt'
            )
            inputs = {k: v.to(self.config["device"]) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_id = torch.argmax(probs, dim=-1).item()
                confidence = float(probs.max().item())
                
                # Get all probabilities
                prob_dict = {}
                for idx, label in self.sentiment_id_to_label.items():
                    prob_dict[label] = float(probs[0][idx].item())
            
            predicted_sentiment = self.sentiment_id_to_label.get(pred_id, "Neutral")
            return predicted_sentiment, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            return "Neutral", 1.0, {"Negative": 0.0, "Neutral": 1.0, "Positive": 0.0}

# ================================
# FastAPI Application
# ================================

app = FastAPI(
    title="Mental Health Support API",
    description="API for emotion detection and sentiment analysis using fine-tuned BERT models",
    version="1.0.0"
)

# Initialize models
try:
    model_handler = ModelConfig()
except Exception as e:
    logger.error(f"Failed to initialize models: {e}")
    model_handler = None

# ================================
# API Routes
# ================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Mental Health Support API",
        "version": "1.0.0",
        "description": "API for emotion detection and sentiment analysis",
        "endpoints": "/docs for Swagger UI documentation"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_status = "unhealthy"
    if model_handler:
        if model_handler.emotion_model_loaded and model_handler.sentiment_model_loaded:
            models_status = "healthy"
        elif model_handler.emotion_model_loaded or model_handler.sentiment_model_loaded:
            models_status = "partial"
        else:
            models_status = "unhealthy"
    
    return HealthResponse(
        status=models_status,
        emotion_model_loaded=model_handler.emotion_model_loaded if model_handler else False,
        sentiment_model_loaded=model_handler.sentiment_model_loaded if model_handler else False,
        timestamp=datetime.now().isoformat()
    )

@app.post("/analyze/emotion", response_model=EmotionResponse)
async def analyze_emotion(input_data: TextInput):
    """
    Analyze emotions in text using the fine-tuned BERT emotion model
    
    - **text**: Input text to analyze
    - **max_predictions**: Maximum number of emotion predictions to return (default: 3)
    """
    if not model_handler or not model_handler.emotion_model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Emotion model is not available due to insufficient memory. Please try again later or use the sentiment analysis endpoint."
        )
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        emotions, scores = model_handler.predict_emotion_top_n(
            input_data.text, 
            input_data.max_predictions
        )
        
        # Format response
        emotion_results = []
        for emotion, score in zip(emotions, scores):
            emotion_results.append(EmotionPrediction(emotion=emotion, confidence=score))
        
        return EmotionResponse(
            emotions=emotion_results,
            primary_emotion=emotions[0],
            confidence=scores[0],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing emotion: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing emotion: {str(e)}")

@app.post("/analyze/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment in text using the fine-tuned DistilBERT sentiment model
    
    - **text**: Input text to analyze
    """
    if not model_handler or not model_handler.sentiment_model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Sentiment model is not available due to insufficient memory. Please try again later or use the emotion analysis endpoint."
        )
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        sentiment, confidence, probabilities = model_handler.predict_sentiment(input_data.text)
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.post("/analyze/combined", response_model=CombinedAnalysisResponse)
async def analyze_combined(input_data: TextInput):
    """
    Perform both emotion and sentiment analysis on text
    
    - **text**: Input text to analyze
    - **max_predictions**: Maximum number of emotion predictions to return (default: 3)
    """
    if not model_handler:
        raise HTTPException(status_code=503, detail="Models not available")
    
    # Check if at least one model is loaded
    if not model_handler.emotion_model_loaded and not model_handler.sentiment_model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Both models are unavailable due to insufficient memory. Please try again later."
        )
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty")
    
    try:
        # Get emotion predictions (if model is available)
        if model_handler.emotion_model_loaded:
            emotions, emotion_scores = model_handler.predict_emotion_top_n(
                input_data.text, 
                input_data.max_predictions
            )
            
            # Format emotion results
            emotion_results = []
            for emotion, score in zip(emotions, emotion_scores):
                emotion_results.append(EmotionPrediction(emotion=emotion, confidence=score))
        else:
            # Return default values when emotion model unavailable
            emotion_results = [EmotionPrediction(emotion="Unavailable", confidence=0.0)]
            emotions = ["Unavailable"]
            emotion_scores = [0.0]
        
        # Get sentiment prediction (if model is available)
        if model_handler.sentiment_model_loaded:
            sentiment, sentiment_confidence, _ = model_handler.predict_sentiment(input_data.text)
        else:
            # Return default values when sentiment model unavailable
            sentiment = "Unavailable"
            sentiment_confidence = 0.0
        
        return CombinedAnalysisResponse(
            text=input_data.text,
            emotions=emotion_results,
            primary_emotion=emotions[0],
            sentiment=sentiment,
            sentiment_confidence=sentiment_confidence,
            analysis_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in combined analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error in combined analysis: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about the loaded models"""
    if not model_handler:
        raise HTTPException(status_code=503, detail="Model handler not initialized")
    
    return {
        "emotion_model_loaded": model_handler.emotion_model_loaded,
        "sentiment_model_loaded": model_handler.sentiment_model_loaded,
        "models_status": "healthy" if (model_handler.emotion_model_loaded and model_handler.sentiment_model_loaded) 
                       else "partial" if (model_handler.emotion_model_loaded or model_handler.sentiment_model_loaded) 
                       else "unhealthy",
        "emotion_model_path": model_handler.config["bert_emotion_model_path"],
        "sentiment_model_path": model_handler.config["bert_sentiment_model_path"],
        "device": model_handler.config["device"],
        "emotion_model_type": "BERT-base-uncased (fine-tuned for emotions)",
        "sentiment_model_type": "DistilBERT-base-uncased (fine-tuned for sentiment)",
        "supported_emotions": "17 emotion classes" if model_handler.emotion_model_loaded else "Unavailable - model not loaded",
        "supported_sentiments": ["Negative", "Neutral", "Positive"] if model_handler.sentiment_model_loaded else "Unavailable - model not loaded"
    }

# ================================
# Error Handlers
# ================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Please check the API documentation at /docs"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please try again later"}

# ================================
# Main execution
# ================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_mental_health_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
