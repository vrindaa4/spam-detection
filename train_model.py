import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Function for text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()
    ...

    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove whitespaces
    text = text.strip()
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back to string
    processed_text = " ".join(tokens)
    
    return processed_text

def download_dataset():
    """Download the SMS Spam Collection dataset from a reliable source"""
    print("Downloading dataset...")
    
    # Try a different reliable source
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    # Create a data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Path to save the zip file
    zip_path = os.path.join(data_dir, "smsspamcollection.zip")
    
    # Path to save the extracted file
    dataset_path = os.path.join(data_dir, "SMSSpamCollection")
    
    # If the dataset already exists, use it
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    # Download the zip file
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded zip file to {zip_path}")
        
        # Extract the zip file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        print(f"Extracted dataset to {data_dir}")
        
        return dataset_path
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Fallback: Create a small sample dataset for testing
        print("Creating a sample dataset for testing...")
        sample_data = [
            ("ham", "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."),
            ("ham", "Ok lar... Joking wif u oni..."),
            ("spam", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"),
            ("ham", "U dun say so early hor... U c already then say..."),
            ("ham", "Nah I don't think he goes to usf, he lives around here though"),
            ("spam", "FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv"),
            ("ham", "Even my brother is not like to speak with me. They treat me like aids patent."),
            ("ham", "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"),
            ("spam", "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."),
            ("spam", "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"),
            ("ham", "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."),
            ("spam", "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"),
            ("ham", "I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."),
            ("ham", "I HAVE A DATE ON SUNDAY WITH WILL!!"),
            ("spam", "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"),
            ("ham", "I've got some cash, we can go to the cinema. What's on?"),
            ("ham", "Ahhh. Work. I vaguely remember that! What does it feel like? Lol"),
            ("ham", "Wait a minute, I'll call after assignment."),
            ("spam", "07732584351 - Rodger Burns - MSG: Our new mobile video service is launching! 4 videos of your choice every month for just £5/month. To cancel send STOP to 87239."),
            ("ham", "Oh k...i'm watching here:)"),
        ]
        
        # Write the sample data to a file
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for label, message in sample_data:
                f.write(f"{label}\t{message}\n")
        
        print(f"Created sample dataset at {dataset_path}")
        return dataset_path

def train_and_save_model():
    print("Training model... This may take a moment.")
    
    # Get the dataset path
    dataset_path = download_dataset()
    
    # Load the SMS Spam Collection dataset
    try:
        df = pd.read_csv(dataset_path, sep='\t', names=['label', 'message'], encoding='latin-1')
    except Exception as e:
        print(f"Error reading dataset: {e}")
        print("Trying with different encoding...")
        df = pd.read_csv(dataset_path, sep='\t', names=['label', 'message'], encoding='utf-8')
    
    print(f"Dataset loaded with {len(df)} messages")
    
    # Preprocess the data
    print("Preprocessing messages...")
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    # Convert labels to binary values
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df = df.dropna(subset=['label'])

    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_message'], df['label'], test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Vectorize the text data
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    # Train the model
    print("Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vect, y_train)
    
    # Save the model and vectorizer
    print("Saving model and vectorizer to disk...")
    pickle.dump(model, open('spam_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    
    # Evaluate the model
    y_pred = model.predict(X_test_vect)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model trained successfully with accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    
    print("Model and vectorizer saved as 'spam_model.pkl' and 'vectorizer.pkl'")

if __name__ == "__main__":
    train_and_save_model()

