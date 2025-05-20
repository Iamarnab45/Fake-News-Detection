import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.fake_news_model import FakeNewsModel
from utils.logger import app_logger
from utils.nlp_features import NLPFeatures
import os
from pathlib import Path

def load_dataset():
    """
    Load the fake and real news dataset.
    
    Returns:
        DataFrame containing the combined dataset
    """
    try:
        # Load real news
        app_logger.info("Loading real news dataset...")
        real_news = pd.read_csv('True.csv')
        real_news['label'] = 0  # 0 for real news
        
        # Load fake news
        app_logger.info("Loading fake news dataset...")
        fake_news = pd.read_csv('Fake.csv')
        fake_news['label'] = 1  # 1 for fake news
        
        # Combine datasets
        df = pd.concat([real_news, fake_news], ignore_index=True)
        
        # Combine title and text
        df['full_text'] = df['title'] + ' ' + df['text']
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        app_logger.info(f"Dataset loaded successfully. Total samples: {len(df)}")
        app_logger.info(f"Real news samples: {len(real_news)}")
        app_logger.info(f"Fake news samples: {len(fake_news)}")
        
        return df
        
    except Exception as e:
        app_logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
        raise

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Remove any rows with missing values
        df = df.dropna(subset=['full_text', 'label'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['full_text'])
        
        # Remove very short texts (less than 50 characters)
        df = df[df['full_text'].str.len() > 50]
        
        app_logger.info(f"After preprocessing: {len(df)} samples remaining")
        return df
        
    except Exception as e:
        app_logger.error(f"Error preprocessing dataset: {str(e)}", exc_info=True)
        raise

def main():
    """Main training function"""
    try:
        # Create necessary directories
        os.makedirs('models/saved', exist_ok=True)
        
        # Load and preprocess data
        df = load_dataset()
        df = preprocess_dataset(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['full_text'],
            df['label'],
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        app_logger.info(f"Training set size: {len(X_train)}")
        app_logger.info(f"Test set size: {len(X_test)}")
        
        # Initialize and train model
        app_logger.info("Initializing model...")
        model = FakeNewsModel()
        
        app_logger.info("Training model...")
        train_metrics = model.train(X_train.tolist(), y_train.tolist())
        app_logger.info(f"Training metrics: {train_metrics}")
        
        # Evaluate model
        app_logger.info("Evaluating model...")
        eval_metrics = model.evaluate(X_test.tolist(), y_test.tolist())
        app_logger.info(f"Evaluation metrics: {eval_metrics}")
        
        # Test with some examples
        test_examples = [
            "Scientists discover new species in Amazon rainforest",
            "Aliens make contact with Earth government, media blackout ordered"
        ]
        
        app_logger.info("\nTesting model with examples:")
        for example in test_examples:
            is_fake, confidence, explanation = model.predict(example)
            app_logger.info(f"\nText: {example}")
            app_logger.info(f"Prediction: {explanation}")
        
        app_logger.info("\nModel training and evaluation completed successfully!")
        
    except Exception as e:
        app_logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 