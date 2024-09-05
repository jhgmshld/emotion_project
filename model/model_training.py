import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import stanza
import re
import nltk
from nltk.corpus import stopwords
from joblib import dump, load
# Download and initialize Stanza
stanza.download('hebrew')#מתקינים את מודל של השפה העברית
nlp = stanza.Pipeline('hebrew', processors='tokenize,mwt,pos,lemma')# מאתחלים את הפיף לאין של השפה העברית ו מציינים את הפעולות שרוצים להשתמש

# Hebrew stopwords
nltk.download('stopwords')# מורידים את הסתוף וורדז
hebrew_stopwords = set(stopwords.words('hebrew'))# בוחרים את הסתוף וורדז של השפה העברית

def preprocess_text(text):
    # Tokenize and lemmatize using Stanza
    doc = nlp(text)# מכינים את המילים לשימוש במודל ,מחלק את הטקסט למשפטים ו מילים
    tokens = [word.lemma.lower() for sent in doc.sentences for word in sent.words]# מפעילים על כל מילה פונקציה שמחלקת אותם ל(שם,פועל,אות ) ,טאגגינג ו לימיטאיזיתיון(החזרה להוא בעבר)
    
    # Remove non-Hebrew characters and stopwords, keep "לא"
    cleaned_tokens = [word for word in tokens if re.match('[א-ת]', word) and (word not in hebrew_stopwords or word == 'לא')]# מוחקים את stopword 
    # את כל מה שהוא לא מילה בעברית ו משאירים את ה לא  stopword
    
    return ' '.join(cleaned_tokens)# מחזירים את הטיקסט יחד עם רווחים שמפרידים בין כל מילה ו מילה

# Load the dataset
df = pd.read_csv('./dataset/emotion_data.csv')#מביאים את המידע וממירים אותו ל PD

# Preprocess the sentences
df['processed_sentence'] = df['sentence'].apply(preprocess_text)# מפעילים את הפונקציה על כל משפט שעושה tokinization,stopwords removal,lemmetization,charachter filtering  

# Encode the labels (emotions)
le = LabelEncoder()# inintialize the ecnoder wich converts the labels of the emotions from text to numbers
df['emotion'] = le.fit_transform(df['emotion'])# actually using the encoder to convert the labels of the emotions from text to numbers

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])# split the dataset to train data 80% and test data 20%,random_state=42 to make sure everytime we run the function we get the same answear,stratified the words

# Define pipelines for SVM and Logistic Regression
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),# we convert the text to features wich helps calculate the accuracy of the model and makes it easier for the model to predict
    ('svm', SVC(probability=True, class_weight='balanced'))# we train the svm model to the data we vectorized
])

logistic_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),# we convert the text to features wich helps calculate the accuracy of the model and makes it easier for the model to predict
    ('lr', LogisticRegression(class_weight='balanced'))# we train the logistic regression model to the data we vectorized
])

# Define the Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('svm', svm_pipeline),#A pipeline that includes TF-IDF vectorization and an SVM classifier.
        ('lr', logistic_pipeline)#A pipeline that includes TF-IDF vectorization and a Logistic Regression classifier
    ],
    voting='soft'
)# combines the predictions of multiple machine learning models to improve overall performance,It aggregates the predictions of individual models (called estimators) and makes a final prediction based on a majority vote (hard voting) or based on the average of predicted probabilities (soft voting).
#Soft voting means that the final prediction is based on the average of the predicted probabilities of the individual models
#For example, if SVM predicts a 70% probability of class A and Logistic Regression predicts 80%, the combined model will predict an average probability of 75% for class A.
#This is typically more accurate than hard voting (which just takes the majority class) because it takes into account the confidence of each model in its prediction.

# Hyperparameter tuning (optional)
param_grid = {
    'svm__tfidf__ngram_range': [(1, 1), (1, 2)],# This tests two different n-gram ranges for the TF-IDF vectorizer within the SVM pipeline: unigrams only (1, 1) and both unigrams and bigrams (1, 2).
    'svm__svm__C': [0.1, 1, 10],#This tests three different values for the regularization parameter C in the SVM model. Smaller C values mean stronger regularization, while larger values allow the model to fit more closely to the training data.
    #High C means less regularization (more flexible, risk of overfitting), and Low C means more regularization (''''simpler model'''''''', risk of underfitting).
    'lr__tfidf__ngram_range': [(1, 1), (1, 2)],
    'lr__lr__C': [0.1, 1, 10]
}#Hyperparameters are parameters that are set before the learning process begins, unlike model parameters that are learned from the data.
#GridSearchCV is a method for systematically searching for the best hyperparameters by evaluating all possible combinations from a predefined set of hyperparameters (the grid).


grid_search = GridSearchCV(estimator=voting_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
#cv=5: This specifies 5-fold cross-validation. The dataset is split into 5 parts, and the model is trained on 4 parts and tested on the remaining 1 part. This process is repeated 5 times, and the results are averaged.
#n_jobs=-1: This allows the grid search to use all available CPU cores for computation, speeding up the process.
#verbose=1: This makes the grid search output more detailed information about its progress to the console.
grid_search.fit(train_df['processed_sentence'], train_df['emotion'])


# Get the best model from grid search
best_voting_classifier = grid_search.best_estimator_# we find the best model for prediction

# Predict on the test data
y_pred = best_voting_classifier.predict(test_df['processed_sentence'])# we predict the test data using that model we choose before

# Decode the predictions back to original labels
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(test_df['emotion'])

# Evaluate the model
accuracy = accuracy_score(y_test_labels, y_pred_labels)
report = classification_report(y_test_labels, y_pred_labels, target_names=le.classes_)
#Provides a detailed report of precision, recall, and F1-score for each class.
#Precision: The ratio of correctly predicted positive observations to the total predicted positives.
#Recall (Sensitivity): The ratio of correctly predicted positive observations to all actual positives.
#F1-score: The weighted average of precision and recall.

#This specifies the names of the classes for which the report is generated. le.classes_ contains the original emotion labels.

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

dump(le, './model/label_encoder.joblib')
dump(best_voting_classifier, './model/emotion_model.joblib')
# Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


