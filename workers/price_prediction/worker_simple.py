# Simple Price Prediction Worker - Using reliable sentiment model
# Optimized for RTX 3070 (8GB VRAM)

import torch
import numpy as np
from transformers import pipeline
import logging
from datetime import datetime
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSentimentWorker:
    def __init__(self):
        # Check if CUDA is available
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'CUDA' if self.device == 0 else 'CPU'}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Use a standard sentiment analysis model that works well
        logger.info("Loading sentiment analysis model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device
        )
        logger.info("Model loaded successfully!")
    
    def analyze_text(self, text):
        """Analyze sentiment of financial text"""
        try:
            # Get sentiment
            result = self.sentiment_pipeline(text, truncation=True, max_length=512)[0]
            
            # Map to financial terms
            sentiment_map = {
                'POSITIVE': 'bullish',
                'NEGATIVE': 'bearish'
            }
            
            sentiment = sentiment_map.get(result['label'], 'neutral')
            
            return {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": sentiment,
                "confidence": float(result['score']),
                "raw_result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return None
    
    def simulate_price_impact(self, sentiment_data):
        """Simulate price impact based on sentiment"""
        if not sentiment_data:
            return None
            
        # Simple simulation
        impact_factors = {
            'bullish': 1.5,   # 1.5% base impact
            'bearish': -1.5,  # -1.5% base impact
            'neutral': 0.0
        }
        
        base_impact = impact_factors.get(sentiment_data['sentiment'], 0)
        confidence = sentiment_data['confidence']
        
        # Calculate final impact
        final_impact = base_impact * confidence
        
        return {
            "sentiment": sentiment_data['sentiment'],
            "confidence": confidence,
            "price_impact_percent": round(final_impact, 2),
            "base_price": 45000,  # Example BTC price
            "predicted_change": round(45000 * final_impact / 100, 2)
        }
    
    def run_demo(self):
        """Run demonstration with various scenarios"""
        logger.info("="*60)
        logger.info("Starting Sentiment Analysis Demo")
        logger.info("="*60)
        
        # Test scenarios
        scenarios = [
            "Bitcoin surges to new all-time high amid institutional buying frenzy",
            "Cryptocurrency market crashes following regulatory crackdown",
            "Bitcoin price remains stable as investors await Fed decision",
            "Major banks announce cryptocurrency adoption plans",
            "Security breach at exchange causes market panic"
        ]
        
        results = []
        
        # Test GPU performance first
        if torch.cuda.is_available():
            logger.info("\n--- GPU Performance Test ---")
            self._test_gpu_performance()
        
        logger.info("\n--- Running Sentiment Analysis ---")
        
        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"\nScenario {i}:")
            logger.info(f"Text: {scenario}")
            
            # Measure inference time
            start_time = time.time()
            sentiment_result = self.analyze_text(scenario)
            inference_time = time.time() - start_time
            
            if sentiment_result:
                logger.info(f"Sentiment: {sentiment_result['sentiment'].upper()}")
                logger.info(f"Confidence: {sentiment_result['confidence']:.2%}")
                logger.info(f"Inference time: {inference_time:.3f}s")
                
                # Calculate price impact
                impact = self.simulate_price_impact(sentiment_result)
                if impact:
                    logger.info(f"Price Impact: {impact['price_impact_percent']:+.2f}%")
                    logger.info(f"Predicted Change: ${impact['predicted_change']:+,.2f}")
                
                results.append({
                    "scenario": i,
                    "text": scenario,
                    "sentiment": sentiment_result,
                    "impact": impact,
                    "inference_time": inference_time
                })
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _test_gpu_performance(self):
        """Test GPU performance"""
        try:
            # Matrix multiplication test
            size = 1024
            a = torch.randn(size, size).cuda()
            b = torch.randn(size, size).cuda()
            
            # Warmup
            for _ in range(10):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            
            # Benchmark
            iterations = 100
            start = time.time()
            for _ in range(iterations):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            tflops = (2 * size**3 * iterations) / (elapsed * 1e12)
            logger.info(f"GPU Performance: {tflops:.2f} TFLOPS")
            
            # Memory info
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
        except Exception as e:
            logger.error(f"GPU test error: {e}")
    
    def _save_results(self, results):
        """Save results to JSON file"""
        output_file = "sentiment_analysis_results.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"\nResults saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main entry point"""
    logger.info("Initializing Simple Sentiment Worker...")
    
    # Activate virtual environment if needed
    import sys
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.info("Running in virtual environment")
    
    # Create and run worker
    worker = SimpleSentimentWorker()
    results = worker.run_demo()
    
    logger.info("\n" + "="*60)
    logger.info("Demo completed successfully!")
    logger.info("="*60)
    
    # Summary
    if results:
        bullish_count = sum(1 for r in results if r['sentiment']['sentiment'] == 'bullish')
        bearish_count = sum(1 for r in results if r['sentiment']['sentiment'] == 'bearish')
        avg_confidence = np.mean([r['sentiment']['confidence'] for r in results])
        avg_inference = np.mean([r['inference_time'] for r in results])
        
        logger.info(f"\nSummary:")
        logger.info(f"- Total scenarios: {len(results)}")
        logger.info(f"- Bullish: {bullish_count}, Bearish: {bearish_count}")
        logger.info(f"- Average confidence: {avg_confidence:.2%}")
        logger.info(f"- Average inference time: {avg_inference:.3f}s")

if __name__ == "__main__":
    main()
