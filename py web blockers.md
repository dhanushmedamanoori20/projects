# projectsimport re
import time
import requests
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Define a list of keywords to block
blocked_keywords = ["social-media", "gambling", "adult-content"]

# Define a simple web content classification model
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

# Train the model with sample data (you should have a larger labeled dataset in practice)
# In this example, we use fictional data for demonstration purposes.
websites = ["https://www.example.com", "https://www.gambling-site.com", "https://www.adultsite.com"]
labels = ["Safe", "Blocked", "Blocked"]

model.fit(websites, labels)


# Function to block a website based on content classification
def block_website_by_content(url, model):
    try:
        article = Article(url)
        article.download()
        article.parse()
        content = article.text

        prediction = model.predict([content])[0]

        if prediction == "Blocked":
            print(f"Blocked {url} due to content.")
            return True

    except Exception as e:
        print(f"An error occurred while checking {url}: {e}")

    return False


# Simulate visiting websites
while True:
    website = input("Enter a website URL (or 'exit' to quit): ")

    if website.lower() == "exit":
        break

    if block_website_by_content(website, model):
        print("This website is blocked.")
    else:
        print("Website accessed.")

    time.sleep(1)  # Simulate browsing delay
