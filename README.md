# **Spam Detection in SMS Messages**

This project implements a **Spam Detection** application that leverages **Machine Learning** techniques to classify SMS messages as either **Spam** or **Ham (Not Spam)**. The application utilizes a labeled SMS dataset to preprocess the text, extract meaningful features, and train a machine learning model capable of predicting whether a message is spam or not.

## **Key Features**

- **Text Preprocessing**: Efficient text cleaning techniques, including removal of punctuation, conversion to lowercase, and other preprocessing steps to prepare the data for training.
- **Spam Word Frequency Analysis**: Identification of the most frequent words within spam messages, with data visualizations to highlight key patterns.
- **SMS Classification**: Machine learning classification using **Logistic Regression** to categorize SMS messages as either **spam** or **ham**.
- **Confidence Threshold Adjustment**: User-adjustable confidence threshold to fine-tune the spam classification results and optimize performance.
- **Interactive User Interface**: An intuitive frontend built with **Streamlit**, enabling users to input SMS messages and receive real-time feedback on whether they are spam or ham.
- **Real-Time Model Updates**: The application supports retraining the model with new data, ensuring the model's performance improves over time.

## **Technologies Used**

- **Python**: The primary programming language used to build the application.
- **Scikit-learn**: Used for building, training, and evaluating the machine learning model.
- **Streamlit**: Provides an interactive and user-friendly interface for real-time interaction with the application.
- **Pandas**: Used for efficient data manipulation and analysis.
- **Matplotlib**: For visualizing data, particularly in presenting spam word frequencies.

## **How to Run the Project**

To run the project locally, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/spam-detection-app.git
Install the required dependencies:


pip install -r requirements.txt
Run the Streamlit application:

streamlit run app.py
Open the application in your browser and start classifying SMS messages.

Future Improvements
Additional Classification Models: Integrating other text classification models (e.g., Naive Bayes, Random Forest) to compare performance and improve accuracy.

Multilingual Support: Adding functionality to classify SMS messages in multiple languages.

Batch Processing: Expanding the user interface to allow batch classification and data analysis.

