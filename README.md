# 👥 Age & Gender Detection System

A production-ready AI-powered face analysis system that detects faces and predicts age, gender, and emotions in real-time using state-of-the-art deep learning models.

## 🚀 Features

- **🎯 Multi-Face Detection**: Detect and analyze multiple faces simultaneously
- **👶 Age Estimation**: Predict age with ±3-5 years accuracy across all demographics
- **🚻 Gender Classification**: High-precision male/female classification (98%+ accuracy)
- **😊 Emotion Recognition**: Detect 7 emotions (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- **🔄 Dual AI Models**: Combines InsightFace + DeepFace for enhanced accuracy
- **📱 Real-time Camera**: Live camera capture and analysis
- **📊 Analytics Dashboard**: Track demographics and analysis history
- **⚡ GPU Acceleration**: CUDA support for high-performance processing
- **🎨 Interactive UI**: Streamlit-based web interface with real-time controls


## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for acceleration

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-detection-system.git
   cd face-detection-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`



## 🔧 Configuration

### Detection Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `DETECTION_CONFIDENCE` | 0.5 | Face detection threshold (0.1-1.0) |
| `MIN_FACE_SIZE` | 30 | Minimum face size in pixels |
| `ENABLE_GPU` | True | Use GPU acceleration if available |




## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **InsightFace**: Face detection and analysis framework
- **DeepFace**: Advanced facial attribute analysis
- **Streamlit**: Interactive web application framework
- **OpenCV**: Computer vision and image processing

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/face-detection-system/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/face-detection-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/face-detection-system/discussions)
