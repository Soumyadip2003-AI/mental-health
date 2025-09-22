# 🧠 Mental Health Crisis Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.85+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI-powered system for detecting mental health crises in text, images, and audio using state-of-the-art machine learning models and explainable AI techniques.

## 🌟 Features

### Core Capabilities
- **Multi-modal Analysis**: Text, image, and audio processing
- **Advanced ML Models**: BERT, RoBERTa, DistilBERT, and ensemble methods
- **Real-time Detection**: Fast API with sub-second response times
- **Explainable AI**: LIME and SHAP explanations for transparency
- **Risk Assessment**: Multi-level risk classification (low, medium, high, critical)
- **Crisis Intervention**: Automated alerts and emergency contact integration

### Technical Features
- **Scalable Architecture**: Microservices with Docker and Kubernetes support
- **Advanced Monitoring**: Prometheus, Grafana, and custom metrics
- **Security**: JWT authentication, rate limiting, and input validation
- **Database**: PostgreSQL with Redis caching
- **API Documentation**: Auto-generated OpenAPI/Swagger docs
- **Testing**: Comprehensive test suite with pytest
- **CI/CD**: GitHub Actions for automated testing and deployment

### AI/ML Features
- **Ensemble Learning**: Multiple model voting and stacking
- **Feature Engineering**: Advanced text preprocessing and feature extraction
- **Model Training**: Automated retraining and hyperparameter optimization
- **A/B Testing**: Model comparison and performance tracking
- **Data Augmentation**: Synthetic data generation for training
- **Transfer Learning**: Pre-trained models with fine-tuning

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mental-health-crisis-detector.git
cd mental-health-crisis-detector
```

2. **Set up environment**
```bash
# Copy environment file
cp .env.example .env

# Edit configuration
nano .env
```

3. **Start with Docker Compose**
```bash
# Development environment
docker-compose --profile dev up -d

# Production environment
docker-compose --profile production up -d

# With monitoring
docker-compose --profile production --profile monitoring up -d
```

4. **Access the application**
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Streamlit App**: http://localhost:8001
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

### Manual Installation

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Download models**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

3. **Initialize database**
```bash
python scripts/init_db.py
```

4. **Train models**
```bash
python train.py --generate_data --model_type ensemble --use_bert
```

5. **Start the application**
```bash
# API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Streamlit app
streamlit run app.py
```

## 📖 Usage

### API Usage

#### Analyze Text
```python
import requests

# Single text analysis
response = requests.post("http://localhost:8000/analyze", json={
    "text": "I feel hopeless and want to end it all",
    "confidence_threshold": 0.7,
    "include_explanation": True
})

result = response.json()
print(f"Crisis Probability: {result['crisis_probability']}")
print(f"Risk Level: {result['risk_level']}")
```

#### Batch Analysis
```python
# Multiple texts
response = requests.post("http://localhost:8000/analyze/batch", json={
    "texts": [
        "I'm feeling great today!",
        "I can't go on anymore",
        "Life is wonderful"
    ],
    "confidence_threshold": 0.5
})
```

### Streamlit Interface

1. **Start the Streamlit app**
```bash
streamlit run app.py
```

2. **Use the web interface**
   - Enter text for analysis
   - Upload images for multimodal analysis
   - Adjust confidence thresholds
   - View explanations and insights

### Python SDK

```python
from services.crisis_detector import CrisisDetectionService

# Initialize service
detector = CrisisDetectionService()
await detector.initialize()

# Analyze text
result = await detector.analyze_text(
    text="I feel hopeless and worthless",
    confidence_threshold=0.7,
    include_explanation=True
)

print(f"Risk Level: {result.risk_level}")
print(f"Confidence: {result.confidence_score}")
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   Database      │
│   Frontend      │◄──►│   Backend       │◄──►│   PostgreSQL    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   ML Models     │
                       │   - BERT        │
                       │   - RoBERTa     │
                       │   - Ensemble    │
                       └─────────────────┘
```

### Model Architecture

```
Input Text
    │
    ▼
┌─────────────┐
│ Preprocessing│
│ - Cleaning   │
│ - Tokenizing │
│ - Normalizing│
└─────────────┘
    │
    ▼
┌─────────────┐
│ Feature     │
│ Extraction  │
│ - TF-IDF    │
│ - Embeddings│
│ - Linguistic│
└─────────────┘
    │
    ▼
┌─────────────┐
│ ML Models   │
│ - BERT      │
│ - RoBERTa   │
│ - Ensemble  │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Risk        │
│ Assessment  │
│ - Scoring   │
│ - Ranking   │
│ - Alerts    │
└─────────────┘
```

## 🔧 Configuration

### Environment Variables

```bash
# Application
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/crisis_detector
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30
RATE_LIMIT_PER_MINUTE=60

# Models
USE_BERT=true
USE_SPACY=true
USE_MULTIMODAL=true
CONFIDENCE_THRESHOLD=0.5

# Monitoring
ENABLE_MONITORING=true
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_PORT=9090
```

### Model Configuration

```python
# config.py
MODEL_CONFIG = {
    "bert": {
        "model_name": "bert-base-uncased",
        "max_length": 512,
        "batch_size": 16,
        "learning_rate": 2e-5
    },
    "ensemble": {
        "models": ["bert", "roberta", "distilbert"],
        "weights": [0.4, 0.3, 0.3],
        "voting": "soft"
    }
}
```

## 📊 Monitoring and Observability

### Metrics
- **System Metrics**: CPU, memory, disk usage
- **Application Metrics**: Request rates, response times, error rates
- **ML Metrics**: Model accuracy, prediction latency, feature importance
- **Business Metrics**: Crisis detection rates, false positives/negatives

### Dashboards
- **Grafana**: System and application monitoring
- **Prometheus**: Metrics collection and alerting
- **Custom Dashboard**: Crisis detection analytics

### Alerting
- **System Alerts**: High resource usage, service failures
- **ML Alerts**: Model performance degradation
- **Crisis Alerts**: High-risk detections requiring intervention

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# All tests with coverage
pytest --cov=. --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API and database testing
- **ML Tests**: Model accuracy and performance
- **Load Tests**: System scalability and performance

## 🚀 Deployment

### Docker Deployment
```bash
# Build and run
docker-compose up -d

# Scale services
docker-compose up -d --scale app=3
```

### Kubernetes Deployment
```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl get services
```

### Cloud Deployment
- **AWS**: EKS, RDS, ElastiCache
- **GCP**: GKE, Cloud SQL, Memorystore
- **Azure**: AKS, Azure Database, Redis Cache

## 🔒 Security

### Security Features
- **Authentication**: JWT tokens with refresh
- **Authorization**: Role-based access control
- **Rate Limiting**: Per-user and per-IP limits
- **Input Validation**: Comprehensive data validation
- **Encryption**: Data encryption at rest and in transit
- **Audit Logging**: Complete audit trail

### Security Best Practices
- Regular security updates
- Dependency vulnerability scanning
- Penetration testing
- Security code reviews
- Incident response procedures

## 🤝 Contributing

### Development Setup
```bash
# Fork and clone
git clone https://github.com/yourusername/mental-health-crisis-detector.git
cd mental-health-crisis-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

### Code Style
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For pre-trained transformer models
- **spaCy**: For advanced NLP processing
- **scikit-learn**: For machine learning algorithms
- **FastAPI**: For the high-performance API framework
- **Streamlit**: For the interactive web interface

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/mental-health-crisis-detector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/mental-health-crisis-detector/discussions)
- **Email**: support@crisis-detector.com

## ⚠️ Disclaimer

This system is designed for research and educational purposes. It should not replace professional mental health assessment or intervention. Always consult qualified mental health professionals for actual crisis assessment and response.

### Emergency Resources
- **National Suicide Prevention Lifeline**: 988
- **Crisis Text Line**: Text HOME to 741741
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

---

**Made with ❤️ for mental health awareness and crisis prevention**
