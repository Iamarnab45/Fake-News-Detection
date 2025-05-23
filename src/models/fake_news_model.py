import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any
import joblib
import os
from pathlib import Path

class FakeNewsModel:
    def __init__(self, model_path: str = "../models/saved/fake_news_model.joblib", base_dir: Path | None = None):
        """
        Initialize the fake news detection model.
        
        Args:
            model_path: Path to save/load the trained model relative to base_dir
            base_dir: The base directory for resolving model_path (e.g., project root)
        """
        if base_dir:
            self.model_path = str(base_dir / model_path)
        else:
            self.model_path = model_path
            
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=42
            ))
        ])
        self.is_trained = False
        
        # Try to load existing model
        self._load_model_if_exists()
    
    def _load_model_if_exists(self):
        """
        Load model if it exists at the specified path
        """
        print(f"Attempting to load model from: {self.model_path}") 
        try:
            if os.path.exists(self.model_path):
                print("Model file found.")
                self.pipeline = joblib.load(self.model_path)
                self.is_trained = True
                print("Model loaded successfully.")
            else:
                print("Model file not found at the specified path.") 
        except Exception as e:
            print(f"Could not load existing model: {str(e)}") 
            self.is_trained = False

    def train(self, texts: list, labels: list) -> Dict[str, float]:
        """
        Train the model on the provided data.
        
        Args:
            texts: List of news article texts
            labels: List of binary labels (0 for real, 1 for fake)
            
        Returns:
            Dictionary containing training metrics
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        
        if not texts:
            raise ValueError("Training data cannot be empty")
        
        # Train the pipeline
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        
        # Calculate training accuracy
        train_accuracy = self.pipeline.score(texts, labels)
        
        # Save the trained model
        self._save_model()
        
        return {
            "train_accuracy": train_accuracy,
            "feature_count": len(self.pipeline.named_steps['tfidf'].get_feature_names_out())
        }

    def predict(self, text: str) -> Tuple[bool, float, str]:
        """
        Predict whether a news article is fake.
        
        Args:
            text: News article text
            
        Returns:
            Tuple of (is_fake, confidence, explanation)
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before making predictions. "
                "Please train the model first or load a trained model."
            )
        
        # Get prediction and probability
        prediction = self.pipeline.predict([text])[0]
        probability = self.pipeline.predict_proba([text])[0]
        confidence = probability[1] if prediction == 1 else probability[0]
        
        # Generate explanation
        explanation = (
            f"This article is classified as {'fake' if prediction == 1 else 'real'} "
            f"with {confidence:.2%} confidence."
        )
        
        return bool(prediction), confidence, explanation

    def _save_model(self):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model saved to: {self.model_path}")     

    def evaluate(self, texts: list, labels: list) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            texts: List of test texts
            labels: List of true labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.pipeline.predict(texts)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
            "f1_score": f1_score(labels, predictions)
        } 