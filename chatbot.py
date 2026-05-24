import nltk
import streamlit as st

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = WordNetLemmatizer()

# Questions
patterns = [
    "hi",
    "hello",
    "hey",
    "bye",
    "goodbye",
    "what is your name",
    "who are you",
    "help",
    "can you help me"
]

# Answers
responses = [
    "Hello!",
    "Hi there!",
    "Hey!",
    "Goodbye!",
    "Bye Bye!",
    "I am an AI chatbot.",
    "I am your virtual assistant.",
    "Sure! Tell me your problem.",
    "Yes, I can help you."
]

# Preprocess text
def preprocess(text):

    tokens = nltk.word_tokenize(text.lower())

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

processed_patterns = [preprocess(pattern) for pattern in patterns]

# Convert text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(processed_patterns)

# Chatbot response function
def chatbot_response(user_input):

    user_input = preprocess(user_input)

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, X)

    index = similarity.argmax()

    score = similarity[0][index]

    if score < 0.3:
        return "Sorry, I don't understand."

    return responses[index]

# Streamlit UI
st.title("🤖 AI Chatbot")

user_input = st.text_input("You:")

if user_input:

    response = chatbot_response(user_input)

    st.write("Bot:", response)