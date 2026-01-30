from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords


# ===== DOWNLOAD STOPWORDS 
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english")) 

# ===== CLEANING FUNCTION =====
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z\s]', "", text)  # Remove special characters
    words = text.split()
    words = [w for w in words if w not in stop_words]  # Remove stopwords
    return " ".join(words)

# ===== LOAD MODELS =====
try:
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    logreg_model = joblib.load("logreg_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    svm_model = joblib.load("svm_model.pkl")
    label_map = {"CA": "CA", "NCA": "NCA"}
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# ===== FLASK APP =====
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("comment", "")

        if not text.strip():
            return render_template(
                "index.html",
                prediction=False,
                error="Please enter a comment!"
            )

        # STEP 1: Clean the input 
        cleaned_text = clean_text(text)
        
        # STEP 2: Transform using TF-IDF
        text_vec = tfidf.transform([cleaned_text])

        # STEP 3: Get predictions
        pred1 = logreg_model.predict(text_vec)[0]
        pred2 = rf_model.predict(text_vec)[0]
        pred3 = svm_model.predict(text_vec)[0]

        # STEP 4: Map to readable labels
        pred1_label = label_map.get(pred1, pred1)
        pred2_label = label_map.get(pred2, pred2)
        pred3_label = label_map.get(pred3, pred3)

        return render_template(
            "index.html",
            prediction=True,
            comment=text,
            pred1=pred1_label,
            pred2=pred2_label,
            pred3=pred3_label
        )

    return render_template("index.html", prediction=False)

if __name__ == "__main__":
    app.run(debug=True)


