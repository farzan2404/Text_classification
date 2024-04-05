import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Load the training data
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med','comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.space',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.os.ms-windows.misc']
news_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
news_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

text_clf = Pipeline([('vect', TfidfVectorizer()),
                    ('clf', MultinomialNB())])
text_clf.fit(news_train.data, news_train.target)

# Streamlit UI
st.title("TopicTagger - A News Category Predition System.")

# Text input for user to provide the news text
news_text = st.text_area("Enter your news text here:")

if st.button("Classify"):
    if news_text:
        # Predict the category
        predicted = text_clf.predict([news_text])

        # Map the predicted category ID to category name
        predicted_category_name = news_test.target_names[predicted[0]]

        st.write("Predicted Category:", predicted_category_name)
    else:
        st.warning("Please enter a news text for classification.")

st.write("Sample results: ")
additional_test_data = [
    "I'm selling my old computer monitor online.",
    "I'm considering buying a new motorcycle.",
#     "Advancements in medical science offer hope for the future.",
#     "Mac hardware enthusiasts are always on the cutting edge.",
#     "Experience a world of creativity through computer graphics.",
#     "Find great deals on items for sale in your neighborhood.",
#     "Rev up your engines for an adrenaline-pumping auto show.",
#     "The open road beckons to motorcycle enthusiasts.",
#     "Baseball enthusiasts celebrate America's pastime.",
#     "Hockey fans unite in the spirit of competition and sportsmanship.",
#     "The world of encryption and secure communication.",
#     "Electronics continue to reshape our daily lives.",
#     "Explore the mysteries of the universe and beyond.",
#     "Gun control debates stir the nation's political landscape.",
#     "Middle East politics and diplomacy in focus.",
#     "Discussing current political events and controversies.",
#     "Promoting interfaith dialogue for peaceful coexistence.",
#     "Stay up-to-date with the latest PC hardware trends.",
#     "Discover the features and functionality of Windows OS.",
]

for data in additional_test_data:
    predicted = text_clf.predict([data])
    predicted_category_name = news_test.target_names[predicted[0]]
    st.write(f"Text: {data} | Predicted Category: {predicted_category_name}")
