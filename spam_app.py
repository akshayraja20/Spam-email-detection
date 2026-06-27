import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📩 Spam Email Detection System")

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Extra spam examples
extra_texts = [
    'I am a financial consultant with privately owned funds for long term investment ROI',
    'My client intends to invest funds in your project guaranteed returns per annum ASAP',
    'Dear friend I am a lawyer with a business proposal for you please reply urgently',
    'I have funds to invest in your country guaranteed profit please contact me ASAP',
    'you are a winner claim your prize now', 'you won a free prize click here',
    'free recharge offer grab now', 'free recharge', 'free recharge today',
    'free recharge limited offer', 'free recharge click now', 'free recharge win',
    'claim your reward today free', 'claim your bonus now limited offer',
    'special offer click now win prize', 'winner selected claim free gift now',
    'free offer win money now click', 'congratulations you won free recharge',
    'bonus cash reward claim now free', 'you have won free gift claim now',
    'offer limited time grab now free', 'free prize offer click now',
    'bonus offer free win now', 'recharge free offer win',
]
extra = pd.DataFrame({'label': [1]*len(extra_texts), 'text': extra_texts})
df = pd.concat([df, extra], ignore_index=True)

# TF-IDF + Naive Bayes
cv = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
X_vec = cv.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB(alpha=0.01)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Dataset info
st.subheader("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Emails", len(df))
col2.metric("Spam Emails", int(df["label"].sum()))
col3.metric("Model Accuracy", f"{acc*100:.2f}%")

st.write("---")

# Check email
st.subheader("🔍 Check Your Email")
user_input = st.text_area("Enter email text here:")

if st.button("Check Spam"):
    if user_input.strip():
        vector = cv.transform([user_input])
        result = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]

        spam_conf = round(proba[1] * 100, 2)
        ham_conf  = round(proba[0] * 100, 2)

        if result == 1:
            st.error("🚫 This is a Spam Email")
        else:
            st.success("✅ This is Not a Spam Email")

        st.write(f"**Spam Probability:** {spam_conf}%")
        st.write(f"**Not Spam Probability:** {ham_conf}%")
        st.progress(int(spam_conf))

        st.write("---")
        st.write("📝 **Message Info**")
        st.write(f"- Word Count: {len(user_input.split())}")
        st.write(f"- Character Count: {len(user_input)}")

    else:
        st.warning("⚠️ Please enter some text")
