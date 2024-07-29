from trial import load_model, predict_phishing
import joblib

def test_urls():
    model_path = 'phishing_detection_model.pth'
    scaler_path = 'feature_scaler.pkl'
    
    try:
        loaded_model, loaded_scaler = load_model(model_path, scaler_path)
        
        # Load feature columns (assuming you saved them during training)
        feature_columns = joblib.load('feature_columns.pkl')
        
        while True:
            url = input("Enter a URL to check (or 'quit' to exit): ").strip()
            
            if url.lower() == 'quit':
                print("Exiting the program.")
                break
            
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url
            
            try:
                prediction, probability = predict_phishing(url, loaded_model, loaded_scaler, feature_columns)
                
                if prediction == 1:
                    print(f"The URL {url} is likely a phishing site (probability: {probability:.2f})")
                else:
                    print(f"The URL {url} is likely not a phishing site (probability of being safe: {1-probability:.2f})")
            
            except Exception as e:
                print(f"An error occurred while processing {url}: {str(e)}")
            
            print()  # Add a blank line for readability
    
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")
        print("Please ensure that the model and scaler files exist and that all required libraries are installed.")

if __name__ == "__main__":
    test_urls()
