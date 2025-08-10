#!/usr/bin/env python3
"""
Mental Health Support Chatbot
Uses the FastAPI Mental Health API as backend tools for emotion and sentiment analysis
"""

import requests
import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import re

# Try to import groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file if available
def load_environment():
    """Load environment variables from .env file"""
    if DOTENV_AVAILABLE:
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info("Loaded environment variables from .env file")
            return True
        else:
            logger.info("No .env file found")
    else:
        logger.info("python-dotenv not available - install with: pip install python-dotenv")
    return False

# Load environment on import
load_environment()

class GroqLLMClient:
    """Client for Groq LLM to generate empathetic responses"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self.error_message = None
        
        if not GROQ_AVAILABLE:
            self.error_message = "Groq library not available. Install with: pip install groq"
            logger.warning(self.error_message)
            return
            
        if not self.api_key:
            self.error_message = "GROQ_API_KEY not found in environment or .env file"
            logger.warning(self.error_message)
            return
        
        # Validate API key format
        if not self.api_key.startswith("gsk_"):
            self.error_message = f"Invalid API key format. Expected format: gsk_... but got: {self.api_key[:10]}..."
            logger.warning(self.error_message)
            return
        
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq LLM client initialized successfully with key: {self.api_key[:8]}...{self.api_key[-4:]}")
        except Exception as e:
            self.error_message = f"Failed to initialize Groq client: {e}"
            logger.error(self.error_message)
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Groq client is available"""
        return self.client is not None
    
    def get_status_message(self) -> str:
        """Get detailed status message"""
        if self.is_available():
            return f"✅ Connected (Key: {self.api_key[:8]}...{self.api_key[-4:]})"
        else:
            return f"❌ {self.error_message or 'Not available'}"
    
    def generate_response(self, user_input: str, emotion_analysis: Dict, sentiment_analysis: Dict, 
                         conversation_context: str = "", user_name: str = "") -> str:
        """Generate empathetic response using Groq LLM"""
        
        if not self.is_available():
            return self._fallback_response(emotion_analysis, sentiment_analysis)
        
        # Prepare the prompt with analysis results
        prompt = self._create_prompt(user_input, emotion_analysis, sentiment_analysis, 
                                   conversation_context, user_name)
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-8b-8192",  # Using Llama 3 8B model
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            
            generated_response = response.choices[0].message.content.strip()
            logger.info(f"Generated response using Groq LLM")
            return generated_response
            
        except Exception as e:
            logger.error(f"Error generating response with Groq: {e}")
            return self._fallback_response(emotion_analysis, sentiment_analysis)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are a compassionate and empathetic mental health support chatbot. Your role is to:

1. Provide supportive, non-judgmental responses
2. Acknowledge and validate the user's emotions
3. Use the emotion and sentiment analysis to craft appropriate responses
4. Encourage healthy coping strategies when appropriate
5. Maintain professional boundaries (you're supportive, not a therapist)
6. Be concise but warm in your responses (2-3 sentences typically)
7. Never provide medical advice or diagnose conditions

Guidelines:
- Always acknowledge the user's feelings
- Use empathetic language
- Offer hope and support
- Ask follow-up questions to encourage sharing
- If crisis indicators are present, prioritize safety resources

Remember: You're here to listen, validate, and support - not to fix or diagnose."""
    
    def _create_prompt(self, user_input: str, emotion_analysis: Dict, sentiment_analysis: Dict,
                      conversation_context: str, user_name: str) -> str:
        """Create a detailed prompt for the LLM"""
        
        # Extract analysis data
        primary_emotion = emotion_analysis.get("primary_emotion", "unknown")
        emotion_confidence = emotion_analysis.get("confidence", 0)
        all_emotions = emotion_analysis.get("emotions", [])
        
        sentiment = sentiment_analysis.get("sentiment", "neutral")
        sentiment_confidence = sentiment_analysis.get("sentiment_confidence", 0)
        sentiment_probs = sentiment_analysis.get("probabilities", {})
        
        prompt = f"""User Message: "{user_input}"

EMOTIONAL ANALYSIS:
- Primary Emotion: {primary_emotion} (confidence: {emotion_confidence:.2f})
- All Detected Emotions: {', '.join([f"{e['emotion']} ({e['confidence']:.2f})" for e in all_emotions[:3]])}
- Overall Sentiment: {sentiment} (confidence: {sentiment_confidence:.2f})
- Sentiment Breakdown: {', '.join([f"{k}: {v:.2f}" for k, v in sentiment_probs.items()])}

USER INFO:
- Name: {user_name or "Not provided"}
- Conversation Context: {conversation_context or "This is early in the conversation"}

TASK: Generate a compassionate, empathetic response that:
1. Acknowledges the detected emotions ({primary_emotion}) and sentiment ({sentiment})
2. Validates their feelings as normal and understandable
3. Provides gentle support and encouragement
4. Asks a thoughtful follow-up question to continue the conversation

Keep the response natural, warm, and 2-3 sentences long. Use the person's name if provided."""
        
        return prompt
    
    def _fallback_response(self, emotion_analysis: Dict, sentiment_analysis: Dict) -> str:
        """Simple fallback response when Groq is not available"""
        primary_emotion = emotion_analysis.get("primary_emotion", "unknown")
        
        return f"I understand you're experiencing {primary_emotion} feelings. I'm here to listen and support you. Would you like to share more about what's on your mind?"

class MentalHealthAPIClient:
    """Client for communicating with the Mental Health API"""
    
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

class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self.user_emotions = []
        self.user_sentiment_trend = []
    
    def add_message(self, role: str, content: str, analysis: Optional[Dict] = None):
        """Add a message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis
        }
        
        self.history.append(message)
        
        # Keep only recent history
        if len(self.history) > self.max_history * 2:  # *2 for user + bot messages
            self.history = self.history[-self.max_history * 2:]
        
        # Track user emotional patterns
        if role == "user" and analysis:
            if "primary_emotion" in analysis:
                self.user_emotions.append(analysis["primary_emotion"])
            if "sentiment" in analysis:
                self.user_sentiment_trend.append(analysis["sentiment"])
    
    def get_recent_emotions(self, count: int = 5) -> List[str]:
        """Get recent user emotions"""
        return self.user_emotions[-count:] if self.user_emotions else []
    
    def get_sentiment_trend(self, count: int = 5) -> List[str]:
        """Get recent sentiment trend"""
        return self.user_sentiment_trend[-count:] if self.user_sentiment_trend else []
    
    def get_conversation_summary(self) -> str:
        """Get a summary of recent conversation"""
        if not self.history:
            return "No conversation history"
        
        recent_messages = self.history[-6:]  # Last 3 exchanges
        summary = []
        
        for msg in recent_messages:
            if msg["role"] == "user":
                summary.append(f"User: {msg['content'][:100]}...")
            else:
                summary.append(f"Bot: {msg['content'][:100]}...")
        
        return "\n".join(summary)

class MentalHealthChatbot:
    """Main chatbot class with integrated mental health analysis tools and LLM responses"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", groq_api_key: str = None):
        self.api_client = MentalHealthAPIClient(api_base_url)
        self.llm_client = GroqLLMClient(groq_api_key)
        self.memory = ConversationMemory()
        self.user_name = None
        self.crisis_keywords = [
            "suicide", "kill myself", "end it all", "can't go on", "want to die",
            "hurt myself", "self harm", "no point", "give up", "hopeless", "better off dead",
            "nothing to live for", "can't take it anymore", "want to disappear"
        ]
        
        # Check API connectivity
        self._check_connections()
    
    def _check_connections(self):
        """Check if API and LLM are available"""
        # Check API connection
        health = self.api_client.health_check()
        if "error" in health:
            logger.warning(f"API connection issue: {health['error']}")
            print("⚠️  Mental Health API is not available. Running with limited analysis.")
        else:
            logger.info("API connection successful")
            print("✅ Connected to Mental Health Analysis API")
        
        # Check LLM connection
        if self.llm_client.is_available():
            print("✅ Connected to Groq LLM for response generation")
        else:
            print("❌ Groq LLM not available - REQUIRED for intelligent responses")
            print(f"   Reason: {self.llm_client.error_message}")
            print("   Please set up your GROQ_API_KEY to use this chatbot effectively.")
    
    def _detect_crisis(self, text: str) -> bool:
        """Detect crisis keywords in user input"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crisis_keywords)
    
    def _get_crisis_response(self) -> str:
        """Return crisis intervention response"""
        return """🆘 I'm very concerned about what you're sharing. Your safety and wellbeing are the most important things right now.

If you're having thoughts of self-harm or suicide, please reach out for immediate help:

🚨 IMMEDIATE HELP:
• National Suicide Prevention Lifeline: 988 (US)
• Crisis Text Line: Text HOME to 741741 (US)
• Emergency Services: 911
• International: Visit findahelpline.com

🤝 ONGOING SUPPORT:
• SAMHSA National Helpline: 1-800-662-4357
• Crisis Chat: suicidepreventionlifeline.org/chat
• Trevor Project (LGBTQ+): 1-866-488-7386

You don't have to go through this alone. There are people who care about you and want to help. These feelings can change, and there is hope."""
    
    def _generate_supportive_response(self, user_input: str, analysis: Dict[str, Any]) -> str:
        """Generate a supportive response using Groq LLM"""
        
        if "error" in analysis:
            if self.llm_client.is_available():
                # Even without analysis, we can still use LLM
                empty_analysis = {"primary_emotion": "unknown", "confidence": 0, "emotions": []}
                empty_sentiment = {"sentiment": "neutral", "sentiment_confidence": 0, "probabilities": {}}
                context = self.memory.get_conversation_summary()
                return self.llm_client.generate_response(
                    user_input, empty_analysis, empty_sentiment, context, self.user_name
                )
            else:
                return "I hear you. Sometimes it's hard to put feelings into words. I'm here to support you - would you like to tell me more about what's on your mind?"
        
        # Separate emotion and sentiment data for cleaner LLM input
        emotion_data = {
            "primary_emotion": analysis.get("primary_emotion", "unknown"),
            "confidence": analysis.get("confidence", 0),
            "emotions": analysis.get("emotions", [])
        }
        
        sentiment_data = {
            "sentiment": analysis.get("sentiment", "neutral"),
            "sentiment_confidence": analysis.get("sentiment_confidence", 0),
            "probabilities": analysis.get("probabilities", {})
        }
        
        # Use LLM for response generation
        if self.llm_client.is_available():
            context = self.memory.get_conversation_summary()
            response = self.llm_client.generate_response(
                user_input, emotion_data, sentiment_data, context, self.user_name
            )
            return response
        else:
            # Simple fallback when LLM is not available
            return "I'm here to listen and support you. Unfortunately, my advanced response system isn't available right now, but I still want to help. Can you tell me more about how you're feeling?"
    
    def _get_emotional_insight(self, analysis: Dict[str, Any]) -> str:
        """Provide insights based on emotional analysis"""
        if "error" in analysis:
            return ""
        
        insights = []
        
        # Primary emotion insight
        primary_emotion = analysis.get("primary_emotion", "")
        confidence = analysis.get("confidence", 0)
        
        if confidence > 0.7:
            insights.append(f"💡 I'm quite confident that your primary emotion is {primary_emotion}.")
        elif confidence > 0.5:
            insights.append(f"💭 I sense that {primary_emotion} might be your main emotion right now.")
        
        # Multiple emotions insight
        emotions = analysis.get("emotions", [])
        if len(emotions) > 2:
            emotion_names = [e["emotion"] for e in emotions[:3]]
            insights.append(f"🌈 You seem to be experiencing a mix of emotions: {', '.join(emotion_names)}. It's normal to feel multiple things at once.")
        
        # Sentiment insight
        sentiment = analysis.get("sentiment", "")
        sentiment_confidence = analysis.get("sentiment_confidence", 0)
        
        if sentiment == "positive" and sentiment_confidence > 0.7:
            insights.append("😊 Overall, I'm picking up positive feelings from you.")
        elif sentiment == "negative" and sentiment_confidence > 0.7:
            insights.append("💙 I notice you might be going through a difficult time emotionally.")
        
        return "\n\n" + "\n".join(insights) if insights else ""
    
    def _get_pattern_insight(self) -> str:
        """Provide insights based on conversation patterns"""
        recent_emotions = self.memory.get_recent_emotions(5)
        recent_sentiment = self.memory.get_sentiment_trend(5)
        
        insights = []
        
        # Emotion patterns
        if len(recent_emotions) >= 3:
            if all(emotion in ["sadness", "fear", "anger"] for emotion in recent_emotions[-3:]):
                insights.append("📊 I've noticed you've been experiencing some challenging emotions lately. It might be helpful to talk about coping strategies.")
            elif all(emotion in ["joy", "content", "surprise"] for emotion in recent_emotions[-3:]):
                insights.append("📊 You seem to be in a positive emotional space recently. That's wonderful to see!")
        
        # Sentiment trends
        if len(recent_sentiment) >= 3:
            negative_count = recent_sentiment[-3:].count("negative")
            if negative_count >= 2:
                insights.append("📈 I notice your overall sentiment has been leaning negative. Remember, it's okay to have difficult days, and I'm here to support you.")
        
        return "\n\n" + "\n".join(insights) if insights else ""
    
    def analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input using all available API tools"""
        logger.info(f"Analyzing user input: {user_input[:50]}...")
        
        # Use combined analysis as primary tool
        analysis = self.api_client.analyze_combined(user_input, max_predictions=3)
        
        if "error" not in analysis:
            logger.info(f"Analysis successful - Emotion: {analysis.get('primary_emotion', 'unknown')}, Sentiment: {analysis.get('sentiment', 'unknown')}")
        else:
            logger.warning(f"Analysis failed: {analysis['error']}")
        
        return analysis
    
    def process_message(self, user_input: str) -> str:
        """Process user message and generate response"""
        
        # Check for crisis indicators first
        if self._detect_crisis(user_input):
            crisis_response = self._get_crisis_response()
            self.memory.add_message("user", user_input)
            self.memory.add_message("bot", crisis_response)
            return crisis_response
        
        # Analyze the user's emotional state
        analysis = self.analyze_user_input(user_input)
        
        # Add to memory
        self.memory.add_message("user", user_input, analysis)
        
        # Generate supportive response
        response = self._generate_supportive_response(user_input, analysis)
        
        # Add emotional insights
        emotional_insight = self._get_emotional_insight(analysis)
        if emotional_insight:
            response += emotional_insight
        
        # Add pattern insights (every few messages)
        if len(self.memory.history) % 6 == 0:  # Every 3 exchanges
            pattern_insight = self._get_pattern_insight()
            if pattern_insight:
                response += pattern_insight
        
        # Add to memory
        self.memory.add_message("bot", response)
        
        return response
    
    def get_analysis_summary(self) -> str:
        """Get a summary of recent emotional analysis"""
        recent_emotions = self.memory.get_recent_emotions(5)
        recent_sentiment = self.memory.get_sentiment_trend(5)
        
        if not recent_emotions and not recent_sentiment:
            return "No recent analysis data available."
        
        summary = "📊 **Your Recent Emotional Patterns:**\n\n"
        
        if recent_emotions:
            emotion_counts = {}
            for emotion in recent_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            summary += "**Emotions:** " + ", ".join([f"{emotion} ({count})" for emotion, count in emotion_counts.items()]) + "\n"
        
        if recent_sentiment:
            sentiment_counts = {}
            for sentiment in recent_sentiment:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            summary += "**Sentiment:** " + ", ".join([f"{sentiment} ({count})" for sentiment, count in sentiment_counts.items()]) + "\n"
        
        return summary
    
    def start_conversation(self):
        """Start the interactive chatbot conversation"""
        print("\n" + "="*60)
        print("🤖 Mental Health Support Chatbot")
        print("="*60)
        print("Hi! I'm here to listen and support you. I use AI to understand")
        print("emotions and sentiment to provide intelligent, personalized responses.")
        print("⚠️  Note: This chatbot requires Groq LLM for best experience.")
        print("\nCommands:")
        print("• 'quit' or 'exit' - End conversation")
        print("• 'summary' - View your emotional analysis summary")
        print("• 'help' - Show detailed help and AI model info")
        print("• 'status' - Check AI system status")
        print("• 'clear' - Clear conversation history")
        print("-"*60)
        
        # Get user's name
        while not self.user_name:
            name_input = input("\nWhat would you like me to call you? (optional, press Enter to skip): ").strip()
            if name_input:
                self.user_name = name_input
                print(f"\nNice to meet you, {self.user_name}! How are you feeling today?")
            else:
                print("\nThat's okay! How are you feeling today?")
                break
        
        # Main conversation loop
        while True:
            try:
                user_input = input(f"\n{self.user_name or 'You'}: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    farewell = f"Take care{', ' + self.user_name if self.user_name else ''}! Remember, it's okay to reach out for support when you need it. 💙"
                    print(f"\n🤖: {farewell}")
                    break
                
                elif user_input.lower() == 'summary':
                    summary = self.get_analysis_summary()
                    print(f"\n🤖: {summary}")
                    continue
                
                elif user_input.lower() == 'help':
                    print(f"""
🤖: Here's how I can help you:

• **AI-Powered Responses**: I use Groq's Llama 3 language model for personalized, empathetic responses
• **Emotion Analysis**: I identify emotions in your messages using BERT-based models
• **Sentiment Analysis**: I analyze the overall tone of your messages (positive, negative, neutral)
• **Pattern Recognition**: I track your emotional patterns over our conversation
• **Crisis Detection**: I can recognize if you might need immediate professional help
• **Conversation Memory**: I remember our recent conversations for better context

**AI Models Used:**
• Emotion Detection: BERT-based emotion classification
• Sentiment Analysis: Advanced sentiment detection
• Response Generation: {"Groq LLM (Llama 3)" if self.llm_client.is_available() else "❌ NOT AVAILABLE - Please set GROQ_API_KEY"}

Commands:
• 'summary' - See your recent emotional patterns
• 'clear' - Start fresh with no conversation history
• 'status' - Check AI model connections
• 'quit' - End our conversation

⚠️ Important: This chatbot requires Groq LLM for intelligent responses. Basic fallback responses are minimal.
I'm here to support you, not replace professional help. If you're in crisis, please contact emergency services.
                    """)
                    continue
                
                elif user_input.lower() == 'clear':
                    self.memory = ConversationMemory()
                    print("\n🤖: Conversation history cleared. How can I support you?")
                    continue
                
                elif user_input.lower() == 'status':
                    # Check API status
                    health = self.api_client.health_check()
                    api_status = "✅ Connected" if "error" not in health else f"❌ Disconnected ({health['error']})"
                    
                    # Check LLM status
                    llm_status = self.llm_client.get_status_message()
                    
                    # Get model info if API is available
                    model_info = ""
                    if "error" not in health:
                        models = self.api_client.get_model_info()
                        if "error" not in models:
                            emotion_model = "✅" if models.get("emotion_model_loaded") else "❌"
                            sentiment_model = "✅" if models.get("sentiment_model_loaded") else "❌"
                            model_info = f"\n• Emotion Model: {emotion_model}\n• Sentiment Model: {sentiment_model}"
                    
                    status_msg = f"""📊 **System Status:**
• Analysis API: {api_status}
• Response LLM: {llm_status}{model_info}

{"🟢 All systems operational!" if self.llm_client.is_available() and "error" not in health else "🟡 Limited functionality - LLM required for intelligent responses"}"""
                    
                    print(f"\n🤖: {status_msg}")
                    continue
                
                # Process regular message
                print("\n🤖 (analyzing emotions & generating response...)", end="", flush=True)
                time.sleep(1.0)  # Slightly longer pause for LLM processing
                print("\r" + " "*50 + "\r", end="")  # Clear the analyzing message
                
                response = self.process_message(user_input)
                print(f"🤖: {response}")
                
            except KeyboardInterrupt:
                print(f"\n\n🤖: Goodbye{', ' + self.user_name if self.user_name else ''}! Take care of yourself. 💙")
                break
            except Exception as e:
                logger.error(f"Error in conversation: {e}")
                print("\n🤖: I'm sorry, I encountered an error. Let's continue our conversation.")

def main():
    """Main function to start the chatbot"""
    print("🤖 Initializing Mental Health Support Chatbot with LLM...")
    
    # Load environment variables
    env_loaded = load_environment()
    
    # Check for Groq API key
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"✅ Found Groq API key: {groq_key[:8]}...{groq_key[-4:]}")
    else:
        print("❌ No GROQ_API_KEY found - REQUIRED for intelligent responses!")
        if env_loaded:
            print("   .env file was loaded but no GROQ_API_KEY found in it")
        else:
            print("   Set GROQ_API_KEY environment variable or create .env file")
        print("\n🔑 To get started:")
        print("   1. Visit: https://console.groq.com/keys")
        print("   2. Get your free API key")
        print("   3. Add GROQ_API_KEY=\"your_key_here\" to your .env file")
        print("   Without LLM, responses will be very basic.\n")
    
    try:
        chatbot = MentalHealthChatbot(groq_api_key=groq_key)
        chatbot.start_conversation()
    except Exception as e:
        logger.error(f"Failed to start chatbot: {e}")
        print(f"❌ Error starting chatbot: {e}")
        print("Please ensure:")
        print("1. FastAPI Mental Health API is running on http://localhost:8000")
        print("2. Required packages are installed: pip install groq requests python-dotenv")
        print("3. GROQ_API_KEY is set in environment or .env file")

if __name__ == "__main__":
    main()
