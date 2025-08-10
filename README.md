# Mental Health Support System

A comprehensive AI-powered mental health support system featuring emotion analysis, sentiment detection, and an intelligent chatbot powered by Groq's Llama 3 LLM.

## 🌟 Features

### FastAPI Backend
- **Emotion Analysis**: BERT-based emotion detection with confidence scoring
- **Sentiment Analysis**: Advanced sentiment classification (positive, negative, neutral)
- **Combined Analysis**: Integrated emotion and sentiment analysis in one endpoint
- **Model Management**: Support for both local and remote Hugging Face models
- **Health Monitoring**: API health checks and model status endpoints

### Intelligent Chatbot
- **LLM-Powered Responses**: Uses Groq's Llama 3 for empathetic, personalized responses
- **Crisis Detection**: Automatic detection of crisis keywords with immediate intervention
- **Conversation Memory**: Tracks emotional patterns and conversation context
- **Pattern Recognition**: Identifies emotional trends and provides insights
- **Professional Boundaries**: Clear limitations with appropriate resource referrals

## 🏗️ System Architecture

```
User Input → Crisis Detection → Emotional Analysis → LLM Response Generation
     ↓              ↓                    ↓                    ↓
Conversation → Safety Check → API Analysis → Groq LLM → Enhanced Response
   Memory                                                        ↓
                                                        Pattern Insights
```

## 📋 Requirements

- Python 3.8+
- FastAPI backend running on `localhost:8000`
- Groq API key for LLM responses
- Required Python packages (see requirements.txt)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Create a `.env` file in the project directory:
```
GROQ_API_KEY=gsk_your_api_key_here
```

Get your free Groq API key at: https://console.groq.com/keys

### 3. Start the FastAPI Backend
```bash
python fastapi_mental_health_api.py
```
The API will be available at: http://localhost:8000

### 4. Launch the Chatbot
```bash
python mental_health_chatbot.py
```

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```http
GET /health
```
Returns API status and model information.

#### Emotion Analysis
```http
POST /analyze/emotion
Content-Type: application/json

{
    "text": "I'm feeling really anxious about tomorrow",
    "max_predictions": 3
}
```

#### Sentiment Analysis
```http
POST /analyze/sentiment
Content-Type: application/json

{
    "text": "I'm feeling really anxious about tomorrow"
}
```

#### Combined Analysis
```http
POST /analyze/combined
Content-Type: application/json

{
    "text": "I'm feeling really anxious about tomorrow",
    "max_predictions": 3
}
```

#### Model Information
```http
GET /models/info
```

### Response Format
```json
{
    "text": "I'm feeling really anxious about tomorrow",
    "primary_emotion": "fear",
    "confidence": 0.85,
    "emotions": [
        {"emotion": "fear", "confidence": 0.85},
        {"emotion": "sadness", "confidence": 0.12},
        {"emotion": "neutral", "confidence": 0.03}
    ],
    "sentiment": "negative",
    "sentiment_confidence": 0.78,
    "probabilities": {
        "negative": 0.78,
        "neutral": 0.15,
        "positive": 0.07
    }
}
```

## 🤖 Chatbot Features

### Core Capabilities
- **Emotion Recognition**: Analyzes user emotions with confidence scoring
- **Intelligent Responses**: Groq Llama 3 generates contextual, empathetic replies
- **Crisis Intervention**: Immediate detection and response to crisis situations
- **Memory System**: Remembers conversation history and emotional patterns
- **Pattern Analysis**: Identifies trends and provides insights over time

### Commands
- `help` - Show detailed help and AI model information
- `status` - Check AI system connectivity and model status
- `summary` - View recent emotional analysis patterns
- `clear` - Reset conversation history
- `quit`/`exit` - End conversation

### Crisis Support Resources
The chatbot automatically provides crisis intervention resources when risk indicators are detected:

- **National Suicide Prevention Lifeline**: 988 (US)
- **Crisis Text Line**: Text HOME to 741741 (US)
- **Emergency Services**: 911
- **SAMHSA National Helpline**: 1-800-662-4357

## 🧠 How the Chatbot Thinks

The chatbot operates on a sophisticated multi-layer approach:

1. **Safety First**: Crisis detection overrides all other processing
2. **Emotional Understanding**: Deep analysis using BERT-based models
3. **Contextual Awareness**: Memory of conversation patterns and history
4. **Empathetic Response**: Groq LLM generates compassionate replies
5. **Pattern Recognition**: Long-term emotional trend analysis

For detailed information about the chatbot's thinking process, see `chatbot_thinking_process.txt`.

## 🔧 Configuration

### Model Configuration
The system supports both local and remote Hugging Face models:

```python
# Local models (place in project directory)
EMOTION_MODEL = "./best_model"  # Local emotion model
SENTIMENT_MODEL = "./best_sentiment_model"  # Local sentiment model

# Remote models (downloaded automatically)
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
```

### Environment Variables
- `GROQ_API_KEY`: Required for chatbot LLM responses
- Custom model paths can be set in the API code

## 📁 File Structure

```
Mental Health Support System/
├── README.md                           # This file
├── requirements.txt                    # All dependencies
├── .env                               # Environment variables (create this)
├── chatbot_thinking_process.txt       # Detailed AI thinking documentation
├── fastapi_mental_health_api.py       # FastAPI backend server
├── mental_health_chatbot.py           # Main chatbot application
├── test_api_client.py                 # API testing client
├── best_model/                        # Local emotion model (if used)
├── best_sentiment_model/              # Local sentiment model (if used)
└── logs/                              # Application logs
    ├── chatbot_errors.log
    └── support_referrals.log
```

## 🧪 Testing

### Test the API
```bash
python test_api_client.py
```

### Manual API Testing
```bash
# Health check
curl http://localhost:8000/health

# Emotion analysis
curl -X POST http://localhost:8000/analyze/emotion \
     -H "Content-Type: application/json" \
     -d '{"text": "I am feeling very happy today!", "max_predictions": 3}'

# Sentiment analysis  
curl -X POST http://localhost:8000/analyze/sentiment \
     -H "Content-Type: application/json" \
     -d '{"text": "I am feeling very happy today!"}'
```

## 🚨 Important Notes

### Safety & Ethics
- **Not a replacement for professional help**: This is a support tool, not therapy
- **Crisis situations**: Always prioritizes safety with immediate resource referrals
- **Privacy**: No conversation data is stored permanently
- **Professional boundaries**: Clear limitations on medical advice

### Technical Requirements
- **Groq API Key**: Required for intelligent chatbot responses
- **Internet connection**: Needed for Groq LLM and model downloads
- **Local models**: Optional - place in project directory for offline emotion/sentiment analysis
- **Memory**: At least 4GB RAM recommended for local model inference

### Limitations
- Requires active internet connection for Groq LLM
- Local models need significant disk space (~500MB each)
- Response time depends on API latency and model size
- English language optimized (other languages may have reduced accuracy)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes thoroughly
4. Submit a pull request with detailed description

## 📄 License

This project is intended for educational and research purposes. Please ensure compliance with:
- Groq API terms of service
- Hugging Face model licenses
- Local privacy and data protection regulations

## 🆘 Support & Resources

### Mental Health Resources
- **Crisis Text Line**: Text HOME to 741741
- **National Suicide Prevention Lifeline**: 988
- **SAMHSA National Helpline**: 1-800-662-HELP (4357)

### Technical Support
- Check `logs/` directory for error messages
- Ensure all requirements are installed
- Verify Groq API key is valid
- Confirm FastAPI backend is running

## 🔄 Updates & Maintenance

The system includes:
- Automatic error logging and monitoring
- Model performance tracking
- Conversation pattern analysis
- Regular updates to crisis detection keywords

For system health monitoring, use the `/health` endpoint and chatbot `status` command.

---

**Remember**: This is a support tool designed to complement, not replace, professional mental health care. If you're experiencing a mental health crisis, please contact emergency services or a crisis hotline immediately.
