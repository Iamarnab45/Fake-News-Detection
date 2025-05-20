import requests
import json

def check_news(title, content, source=None):
    """
    Check if a news article is fake or real.
    
    Args:
        title (str): The headline of the news article
        content (str): The main text of the article
        source (str, optional): The website or source of the news
    
    Returns:
        dict: The prediction results
    """
    url = "http://localhost:8000/predict"
    
    data = {
        "title": title,
        "content": content
    }
    if source:
        data["source"] = source
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print("\nAnalysis Results:")
        print("-" * 50)
        print(f"Prediction: {'FAKE' if result['is_fake'] else 'REAL'} news")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Explanation: {result['explanation']}")
        print("\nKey Features:")
        for feature, value in result['features'].items():
            print(f"- {feature}: {value:.2f}")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to the API server. Make sure it's running.")
        print(f"Details: {str(e)}")
        return None

if __name__ == "__main__":
    # Preity Zinta news article
    title = "Preity Zinta slams morphed PICS with Vaibhav Sooryavanshi which showed her hugging the cricketer, calls it 'fake news'"
    content = """
    Preity Zinta, co-owner of Punjab Kings, refuted viral images circulating online that depicted her hugging young cricketer Vaibhav Sooryavanshi. 
    Zinta addressed the morphed images on X, clarifying the news as fake. 
    Meanwhile, she celebrated Punjab Kings' playoff qualification, praising the team's performance. 
    Zinta is also set to return to acting with 'Lahore 1947,' alongside Sunny Deol, produced by Aamir Khan.
    """
    
    check_news(title, content, source="Times of India") 