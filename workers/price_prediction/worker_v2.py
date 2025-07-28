# Price Prediction Worker V2 - Using Pre-trained Financial Sentiment Model
# Optimized for RTX 3070 (8GB VRAM)

import torch
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import os
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialSentimentWorker:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        logger.info(f"Using device: {'cuda' if self.device == 0 else 'cpu'}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
        # Use a pre-trained financial sentiment model
        logger.info("Loading pre-trained financial sentiment model...")
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",  # Financial BERT model
            device=self.device
        )
        logger.info("Model loaded successfully!")
    
    def analyze_market_sentiment(self, text):
        """Analyze financial text sentiment"""
        try:
            # Run sentiment analysis
            result = self.sentiment_analyzer(text, truncation=True, max_length=512)[0]
            
            # Map labels to our sentiment categories
            label_map = {
                'positive': 'bullish',
                'negative': 'bearish',
                'neutral': 'neutral'
            }
            
            sentiment = label_map.get(result['label'].lower(), 'neutral')
            
            return {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": sentiment,
                "confidence": float(result['score']),
                "raw_label": result['label'],
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None
    
    def simulate_price_prediction(self, sentiment_data):
        """Simulate a price prediction based on sentiment"""
        # This is a simple simulation - in production, you'd use more sophisticated models
        sentiment_weights = {
            'bullish': 0.02,   # 2% increase
            'neutral': 0.0,    # No change
            'bearish': -0.02   # 2% decrease
        }
        
        base_price = 45000  # Example: Bitcoin price
        sentiment_impact = sentiment_weights.get(sentiment_data['sentiment'], 0)
        confidence_factor = sentiment_data['confidence']
        
        # Price change based on sentiment and confidence
        price_change = base_price * sentiment_impact * confidence_factor
        predicted_price = base_price + price_change
        
        return {
            "current_price": base_price,
            "predicted_price": round(predicted_price, 2),
            "price_change": round(price_change, 2),
            "percentage_change": round(sentiment_impact * confidence_factor * 100, 2)
        }
    
    def run_analysis(self):
        """Run sentiment analysis on various market scenarios"""
        logger.info("Starting Financial Sentiment Analysis Worker")
        logger.info("=" * 60)
        
        # Test scenarios
        test_scenarios = [
            "Bitcoin rallies to new all-time high as institutional investors pour billions into cryptocurrency market",
            "Federal Reserve announces surprise rate hike causing market volatility and uncertainty",
            "Cryptocurrency market remains stable as traders await regulatory clarity from SEC",
            "Major crypto exchange hacked, billions in user funds at risk causing market panic",
            "Tesla announces additional $1.5 billion Bitcoin purchase, boosting market confidence"
        ]
        
        results = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\nScenario {i}:")
            logger.info(f"Text: {scenario[:80]}...")
            
            # Analyze sentiment
            sentiment_result = self.analyze_market_sentiment(scenario)
            if sentiment_result:
                logger.info(f"Sentiment: {sentiment_result['sentiment'].upper()} (confidence: {sentiment_result['confidence']:.2%})")
                
                # Simulate price prediction
                price_prediction = self.simulate_price_prediction(sentiment_result)
                logger.info(f"Price Impact: {price_prediction['percentage_change']:+.2f}%")
                logger.info(f"Predicted Price: ${price_prediction['predicted_price']:,.2f} (from ${price_prediction['current_price']:,.2f})")
                
                results.append({
                    "scenario": i,
                    "sentiment": sentiment_result,
                    "prediction": price_prediction
                })
        
        # Save results
        output_file = "analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {output_file}")
        
        return results

def main():
    """Main function to run the worker"""
    worker = FinancialSentimentWorker()
    
    # Test GPU performance
    if torch.cuda.is_available():
        logger.info("\n=== GPU Performance Test ===")
        # Simple benchmark
        size = 512
        a = torch.randn(size, size).cuda()
        b = torch.randn(size, size).cuda()
        
        # Warmup
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Measure
        import time
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        gflops = (2 * size**3 * iterations) / (elapsed * 1e9)
        logger.info(f"GPU Performance: {gflops:.2f} GFLOPS")
    
    # Run analysis
    worker.run_analysis()
    
    logger.info("\n=== Worker completed successfully! ===")

if __name__ == "__main__":
    main()
