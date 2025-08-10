#!/usr/bin/env python3
"""
Test client for FastAPI Mental Health Support API
Demonstrates how to use all available endpoints
"""

import requests
import json
from typing import Dict, Any

class MentalHealthAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Health check failed: {e}"}
    
    def analyze_emotion(self, text: str, max_predictions: int = 3) -> Dict[str, Any]:
        """Analyze emotions in text"""
        try:
            payload = {
                "text": text,
                "max_predictions": max_predictions
            }
            response = requests.post(f"{self.base_url}/analyze/emotion", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Emotion analysis failed: {e}"}
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment in text"""
        try:
            payload = {"text": text}
            response = requests.post(f"{self.base_url}/analyze/sentiment", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Sentiment analysis failed: {e}"}
    
    def analyze_combined(self, text: str, max_predictions: int = 3) -> Dict[str, Any]:
        """Perform both emotion and sentiment analysis"""
        try:
            payload = {
                "text": text,
                "max_predictions": max_predictions
            }
            response = requests.post(f"{self.base_url}/analyze/combined", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Combined analysis failed: {e}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        try:
            response = requests.get(f"{self.base_url}/models/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Model info request failed: {e}"}

def main():
    """Test the API with sample inputs"""
    client = MentalHealthAPIClient()
    
    print("Mental Health API Test Client")
    print("=" * 50)
    
    # Health check
    print("\n1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if health.get("status") != "healthy":
        print("API is not healthy. Please check the server.")
        return
    
    # Model info
    print("\n2. Model Information:")
    model_info = client.get_model_info()
    print(json.dumps(model_info, indent=2))
    
    # Test cases
    test_texts = [
        "I'm feeling really anxious about my upcoming exams",
        "Today was an amazing day! I accomplished so much",
        "I feel lonely and don't know what to do",
        "I'm grateful for all the support from my friends",
        "This situation is making me angry and frustrated"
    ]
    
    print("\n3. Testing Emotion Analysis:")
    print("-" * 30)
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        result = client.analyze_emotion(text)
        if "error" not in result:
            print(f"Primary Emotion: {result['primary_emotion']} ({result['confidence']:.3f})")
            print("All Emotions:")
            for emotion_data in result['emotions']:
                print(f"  - {emotion_data['emotion']}: {emotion_data['confidence']:.3f}")
        else:
            print(f"Error: {result['error']}")
    
    print("\n4. Testing Sentiment Analysis:")
    print("-" * 30)
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        result = client.analyze_sentiment(text)
        if "error" not in result:
            print(f"Sentiment: {result['sentiment']} ({result['confidence']:.3f})")
            print("Probabilities:")
            for sentiment, prob in result['probabilities'].items():
                print(f"  - {sentiment}: {prob:.3f}")
        else:
            print(f"Error: {result['error']}")
    
    print("\n5. Testing Combined Analysis:")
    print("-" * 30)
    sample_text = "I'm really excited about my presentation tomorrow, but also a bit nervous"
    print(f"Text: '{sample_text}'")
    result = client.analyze_combined(sample_text)
    if "error" not in result:
        print(f"Primary Emotion: {result['primary_emotion']}")
        print(f"Sentiment: {result['sentiment']} ({result['sentiment_confidence']:.3f})")
        print("All Emotions:")
        for emotion_data in result['emotions']:
            print(f"  - {emotion_data['emotion']}: {emotion_data['confidence']:.3f}")
    else:
        print(f"Error: {result['error']}")
    
    print("\nAPI testing completed!")

if __name__ == "__main__":
    main()
