# Price Prediction Worker - Optimized for RTX 3070
# Example implementation using lightweight models

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PricePredictionWorker:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load model optimized for 8GB VRAM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3  # bearish, neutral, bullish
        ).to(self.device)
        
        # Enable mixed precision for RTX 3070 tensor cores
        self.model.half()
        logger.info(f"Model loaded: {model_name}")
    
    def predict_sentiment(self, text):
        """Predict market sentiment from text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        sentiment_map = {0: "bearish", 1: "neutral", 2: "bullish"}
        predicted_class = predictions.argmax().item()
        confidence = predictions.max().item()
        
        return {
            "sentiment": sentiment_map[predicted_class],
            "confidence": float(confidence),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def run(self):
        """Main worker loop"""
        logger.info("Price Prediction Worker started")
        
        # Example prediction
        test_texts = [
            "Bitcoin shows strong bullish momentum with increasing volume",
            "Market uncertainty continues as traders await Fed decision",
            "Massive sell-off triggered by regulatory concerns"
        ]
        
        for text in test_texts:
            result = self.predict_sentiment(text)
            logger.info(f"Text: {text[:50]}...")
            logger.info(f"Result: {result}")

if __name__ == "__main__":
    worker = PricePredictionWorker()
    worker.run()
