# Fake News Detection Web Application

A machine learning-powered web application that analyzes news articles to detect potential fake news using FastAPI and natural language processing.

## Features

- Real-time news article analysis
- Machine learning-based fake news detection
- User-friendly web interface
- Confidence score and explanation for predictions
- RESTful API endpoints

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git LFS (for handling large model files)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Iamarnab45/Fake-News-Detection.git
cd Fake-News-Detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web application:
```bash
cd src
uvicorn main:app --reload
```

2. Open your web browser and navigate to:
```
http://localhost:8000
```

3. Enter a news article URL in the input field and click "Analyze" to get the prediction.

## API Endpoints

- `POST /analyze`: Analyze a news article URL
  - Request body: `{"url": "news_article_url"}`
  - Response: Prediction result with confidence score and explanation

## Project Structure

```
Fake-News-Detection/
├── src/
│   ├── main.py              # FastAPI application
│   ├── check_news.py        # News analysis logic
│   ├── train_model.py       # Model training script
│   ├── static/             # Static files (CSS, JS)
│   ├── templates/          # HTML templates
│   └── utils/              # Utility functions
├── models/
│   └── saved/             # Saved model files
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Model Information

The application uses a machine learning model trained on a dataset of real and fake news articles. The model analyzes various features of the text to make predictions about the authenticity of news articles.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- FastAPI: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- Scikit-learn: [Scikit-learn Documentation](https://scikit-learn.org/) 