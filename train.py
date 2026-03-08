import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split    
from preprocessing import clean_text

# Load Dataset
df=pd.read_csv('data/spam.csv',encoding='latin-1')
df=df[['v1','v2']]
df.columns=('label','message')

# Clean Text
df['clean_message']=df['message'].apply(clean_text)

# Feature extraction
tfidf=TfidfVectorizer(max_features=3000)
X=tfidf.fit_transform(df['clean_message'])
y=df['label']

# Split 
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

# Train 
model= LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# Saving the model and Vectorizer 
with open('models/spam_model.pkl','wb') as f :
    pickle.dump(model,f)

with open('models/tfidf_vectorizer.pkl','wb') as f:
    pickle.dump(tfidf,f)

print("Model Trained and Saved")
print(f"Training samples:{X_train.shape[0]}")
print(f"Test sample:{X_test.shape[0]}")