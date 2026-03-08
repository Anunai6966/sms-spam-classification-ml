import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from preprocessing import clean_text

# Load dataset
df=pd.read_csv('data/spam.csv',encoding='latin-1')
df=df[['v1','v2']]
df.columns=['label','message']
df['clean_message']=df['message'].apply(clean_text)

# Load model and vectorizer
with open ('models/spam_model.pkl','rb') as f:
    model=pickle.load(f)

with open ('models/tfidf_vectorizer.pkl','rb') as f:
    vectorizer=pickle.load(f)

#recreating same split
X=vectorizer.transform(df['clean_message'])
y=df['label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#evaluation
y_pred=model.predict(X_test)

print('='*60)
print("           MODEL EVALUATION REPORT")
print('='*60)
print(classification_report(y_test,y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))