import pickle
from preprocessing import clean_text

# Load Saved Model
with open ('models/spam_model.pkl','rb') as f:
    model=pickle.load(f)

with open('models/tfidf_vectorizer.pkl','rb') as f:
    vectorizer=pickle.load(f)

def predict(message):
    # prediction
    cleaned=clean_text(message)
    features=vectorizer.transform([cleaned])
    label=model.predict(features)[0]
    confidence=model.predict_proba(features)[0].max()
    return label, round(confidence*100,2)

# Test it
if __name__ == '__main__':
    tests = [
        "Congratulations! You won a free iPhone. Click now!",
        "Hey are you coming to class tomorrow?",
        "WINNER!! Claim your $1000 prize now, call 09061743810",
        "Ok I will meet you at 6pm"
    ]

    for msg in tests:
        label, confidence = predict(msg)
        print(f"[{label.upper()}] {confidence}% — {msg[:50]}")