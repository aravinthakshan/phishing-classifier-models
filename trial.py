import torch
import torch.nn as nn
import joblib

import torch.nn.functional as F  # Import torch.nn.functional as F

import numpy as np
import re
import numpy as np
import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from whois import whois
from datetime import datetime
import dns.resolver

# Define the DenseNet model
class DenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(in_features + i * growth_rate, growth_rate),
                nn.BatchNorm1d(growth_rate),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)

class DenseNet(nn.Module):
    def __init__(self, num_features, num_classes, growth_rate=32, num_layers_per_block=4):
        super(DenseNet, self).__init__()
        self.input_layer = nn.Linear(num_features, growth_rate)
        self.dense_block1 = DenseBlock(growth_rate, growth_rate, num_layers_per_block)
        self.transition_layer1 = nn.Linear(growth_rate + num_layers_per_block * growth_rate, growth_rate)
        self.dense_block2 = DenseBlock(growth_rate, growth_rate, num_layers_per_block)
        self.transition_layer2 = nn.Linear(growth_rate + num_layers_per_block * growth_rate, growth_rate)
        self.output_layer = nn.Linear(growth_rate, num_classes)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dense_block1(x)
        x = F.relu(self.transition_layer1(x))
        x = self.dense_block2(x)
        x = F.relu(self.transition_layer2(x))
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)

# Load the trained model and scaler
def load_model(model_path, scaler_path):
    num_features = 30  # Assuming the number of features is 30
    num_classes = 2
    model = DenseNet(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    scaler = joblib.load(scaler_path)
    
    return model, scaler

# Feature extraction
def extract_features(url):
    features = {}
    try:
        features['having_IPhaving_IP_Address'] = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", urlparse(url).netloc) else -1
    except:
        features['having_IPhaving_IP_Address'] = -1
        
    try:
        features['URLURL_Length'] = 1 if len(url) < 54 else 0 if len(url) >= 54 and len(url) <= 75 else -1
    except:
        features['URLURL_Length'] = -1
        
    try:
        features['Shortining_Service'] = 1 if any(service in url for service in ["bit.ly", "goo.gl", "tinyurl.com", "t.co"]) else -1
    except:
        features['Shortining_Service'] = -1
        
    try:
        features['having_At_Symbol'] = 1 if '@' in url else -1
    except:
        features['having_At_Symbol'] = -1
        
    try:
        features['double_slash_redirecting'] = 1 if url.replace('https://', '').find('//') != -1 else -1
    except:
        features['double_slash_redirecting'] = -1
        
    try:
        features['Prefix_Suffix'] = 1 if '-' in urlparse(url).netloc else -1
    except:
        features['Prefix_Suffix'] = -1
        
    try:
        features['having_Sub_Domain'] = -1 if len(urlparse(url).netloc.split('.')) <= 2 else 0 if len(urlparse(url).netloc.split('.')) == 3 else 1
    except:
        features['having_Sub_Domain'] = -1
        
    try:
        features['SSLfinal_State'] = 1 if requests.get(url, verify=True).ok else -1
    except:
        features['SSLfinal_State'] = -1
        
    try:
        domain_info = whois(urlparse(url).netloc)
        if domain_info.expiration_date and domain_info.creation_date:
            expiration_date = domain_info.expiration_date[0] if isinstance(domain_info.expiration_date, list) else domain_info.expiration_date
            creation_date = domain_info.creation_date[0] if isinstance(domain_info.creation_date, list) else domain_info.creation_date
            registration_length = (expiration_date - creation_date).days
            features['Domain_registeration_length'] = 1 if registration_length > 365 else -1
        else:
            features['Domain_registeration_length'] = -1
    except:
        features['Domain_registeration_length'] = -1
        
    try:
        features['Favicon'] = 1 if requests.get(f"{urlparse(url).scheme}://{urlparse(url).netloc}/favicon.ico").status_code == 200 else -1
    except:
        features['Favicon'] = -1
        
    try:
        features['port'] = 1 if urlparse(url).port in [80, 443, None] else -1
    except:
        features['port'] = -1
        
    try:
        features['HTTPS_token'] = -1 if 'https-' in urlparse(url).netloc else 1
    except:
        features['HTTPS_token'] = -1
        
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        features['Request_URL'] = len([link for link in soup.find_all('link', href=True) if urlparse(link['href']).netloc in url]) / len(soup.find_all('link')) if soup.find_all('link') else 1
    except:
        features['Request_URL'] = -1
        
    try:
        features['URL_of_Anchor'] = len([a for a in soup.find_all('a', href=True) if urlparse(a['href']).netloc in url]) / len(soup.find_all('a')) if soup.find_all('a') else 1
    except:
        features['URL_of_Anchor'] = -1
        
    try:
        features['Links_in_tags'] = len([link for link in soup.find_all(['meta', 'script', 'link'], src=True) if urlparse(link['src']).netloc in url]) / (len(soup.find_all('meta')) + len(soup.find_all('script')) + len(soup.find_all('link'))) if (len(soup.find_all('meta')) + len(soup.find_all('script')) + len(soup.find_all('link'))) > 0 else 1
    except:
        features['Links_in_tags'] = -1
        
    try:
        features['SFH'] = len([form for form in soup.find_all('form') if urlparse(form.get('action', '')).netloc in url or form.get('action', '') == '']) / len(soup.find_all('form')) if soup.find_all('form') else 1
    except:
        features['SFH'] = -1
        
    try:
        features['Submitting_to_email'] = 1 if re.search(r'mailto:', r.text) is None else -1
    except:
        features['Submitting_to_email'] = -1
        
    try:
        features['Abnormal_URL'] = 1 if domain_info.get('domain_name', '') else -1
    except:
        features['Abnormal_URL'] = -1
        
    try:
        features['Redirect'] = 0 if len(r.history) <= 1 else 1
    except:
        features['Redirect'] = -1
        
    try:
        features['on_mouseover'] = 1 if re.search(r'onmouseover', r.text) is None else -1
    except:
        features['on_mouseover'] = -1
        
    try:
        features['RightClick'] = 1 if re.search(r'event.button == 2', r.text) is None else -1
    except:
        features['RightClick'] = -1
        
    try:
        features['popUpWidnow'] = 1 if re.search(r'alert\(', r.text) is None else -1
    except:
        features['popUpWidnow'] = -1
        
    try:
        features['Iframe'] = 1 if re.search(r'<iframe', r.text) is None else -1
    except:
        features['Iframe'] = -1
        
    try:
        if domain_info.creation_date:
            creation_date = domain_info.creation_date[0] if isinstance(domain_info.creation_date, list) else domain_info.creation_date
            age = (datetime.now() - creation_date).days
            features['age_of_domain'] = 1 if age > 180 else -1
        else:
            features['age_of_domain'] = -1
    except:
        features['age_of_domain'] = -1
        
    try:
        features['DNSRecord'] = 1 if domain_info else -1
    except:
        features['DNSRecord'] = -1
        
    try:
        features['web_traffic'] = 1 if requests.get(f"http://data.alexa.com/data?cli=10&url={url}").status_code == 200 else -1
    except:
        features['web_traffic'] = -1
        
    try:
        features['Page_Rank'] = 1 if requests.get(f"https://www.checkpagerank.net/index.php?name={urlparse(url).netloc}").status_code == 200 else -1
    except:
        features['Page_Rank'] = -1
        
    try:
        features['Google_Index'] = 1 if '200' in requests.get(f"https://www.google.com/search?q=site:{urlparse(url).netloc}").text else -1
    except:
        features['Google_Index'] = -1
        
    try:
        features['Links_pointing_to_page'] = 1 if requests.get(f"http://data.alexa.com/data?cli=10&url={url}").status_code == 200 else -1
    except:
        features['Links_pointing_to_page'] = -1
        
    try:
        features['Statistical_report'] = 1 if requests.get(f"https://www.phishtank.com/check_phish.php?url={url}").status_code == 200 else -1
    except:
        features['Statistical_report'] = -1
        
    return features

# Predict if the URL is phishing
def predict_phishing(url, model, scaler, feature_columns):
    features = extract_features(url)
    features_list = [features[col] for col in feature_columns]  # Ensure the order matches the training data
    features_scaled = scaler.transform([features_list])
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(features_tensor)
        _, predicted = torch.max(output, 1)
        probability = torch.exp(output).max().item()
    
    return predicted.item(), probability
