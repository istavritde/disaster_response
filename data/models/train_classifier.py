import sys
# import libraries
import datetime
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem.wordnet import WordNetLemmatizer

import pickle


from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.ensemble import  RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR
from sklearn.model_selection  import train_test_split,GridSearchCV
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score,accuracy_score,make_scorer,classification_report,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.decomposition import PCA, TruncatedSVD


import seaborn as sns
import matplotlib.pyplot as plt

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(table_name="DisasterResponse_table",con=engine)
    X = df['message']
    y = df.drop(['message','original','genre','id'],axis=1) 
    category_names = y.columns
    return X, y, category_names

def tokenize(text):

    text = re.sub(r'([a-z])([A-Z])',r'\1\. \2',text)
    text = re.sub('\s+', ' ', text) # remove \t and \n
    text = text.translate(str.maketrans('','',string.punctuation))
    text = " ".join([w for w in text.split() if w not in stopwords.words("english")])
    text = " ".join([PorterStemmer().stem(w) for w in text.split()])
    text = " ".join([WordNetLemmatizer().lemmatize(w) for w in text.split()]).lower()
    text = word_tokenize(text)        
                    
    return text


def build_model(clf=RandomForestClassifier(max_depth=100,min_samples_leaf=1,n_estimators=150,random_state=1,n_jobs=-1)):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize,use_idf=True,smooth_idf=True,max_df=0.98,min_df=0.01)
    pipeline = Pipeline([('tfidf_vectorizer',tfidf_vectorizer),('clf',clf)])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred_test, target_names=category_names))


def save_model(model, model_filepath):

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()