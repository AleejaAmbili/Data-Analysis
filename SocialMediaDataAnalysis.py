#!/usr/bin/env python
# coding: utf-8

# # Clean & Analyze Social Media

# ## Introduction
# 
# Social media has become a ubiquitous part of modern life, with platforms such as Instagram, Twitter, and Facebook serving as essential communication channels. Social media data sets are vast and complex, making analysis a challenging task for businesses and researchers alike. In this project, we explore a simulated social media, for example Tweets, data set to understand trends in likes across different categories.
# 
# ## Prerequisites
# 
# To follow along with this project, you should have a basic understanding of Python programming and data analysis concepts. In addition, you may want to use the following packages in your Python environment:
# 
# - pandas
# - Matplotlib
# - ...
# 
# These packages should already be installed in Coursera's Jupyter Notebook environment, however if you'd like to install additional packages that are not included in this environment or are working off platform you can install additional packages using `!pip install packagename` within a notebook cell such as:
# 
# - `!pip install pandas`
# - `!pip install matplotlib`
# 
# ## Project Scope
# 
# The objective of this project is to analyze tweets (or other social media data) and gain insights into user engagement. We will explore the data set using visualization techniques to understand the distribution of likes across different categories. Finally, we will analyze the data to draw conclusions about the most popular categories and the overall engagement on the platform.
# 
# ## Step 1: Importing Required Libraries
# 
# As the name suggests, the first step is to import all the necessary libraries that will be used in the project. In this case, we need pandas, numpy, matplotlib, seaborn, and random libraries.
# 
# Pandas is a library used for data manipulation and analysis. Numpy is a library used for numerical computations. Matplotlib is a library used for data visualization. Seaborn is a library used for statistical data visualization. Random is a library used to generate random numbers.

# In[1]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install wordcloud')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


# In[2]:


# Generating random data

# generating random likes for posts

likes = np.random.randint(50,1000,20)
retweets = np.random.randint(10,500,20)
print("Random Likes",likes)
print("Random Retweets",retweets)


# In[3]:


# Generating random categories

categories = np.random.choice(['Tech','Fashion','Food','Travel','Health','Food','Family','Culture'],20)
print(categories)


# In[4]:


#Generating random Texts

texts = [
    "New AI tool released! #Tech",
    "Try this delicious pasta recipe! #Foodie",
    "Best travel destinations to explore! #Wanderlust",
    "Morning workouts boost energy! #Fitness",
    "Art brings people together! #Culture",
    "This song is amazing! #NowPlaying",
    "Latest fashion trends to try! #Style",
    "Healthy meal ideas for you! #Food",
    "Spending quality time with family! #Love",
    "Tech innovations of 2024! #FutureTech"
]

random_texts = [random.choice(texts) for _ in range(20)]
print(random_texts)


# In[5]:



# Manually assigning text to match each category
data = {
    "post_id": range(1, 21),
    "category": [
        "Culture", "Tech", "Travel", "Fashion", "Family",
        "Food", "Health", "Music", "Fitness", "Beauty",
        "Culture", "Tech", "Travel", "Fashion", "Family",
        "Food", "Health", "Music", "Fitness", "Beauty"
    ],
    "text": [
        "Art brings people together! #Culture",
        "New AI technology is changing the world! #Tech",
        "Top 10 travel destinations this year! #Travel",
        "New fashion trends of 2024! #Style",
        "Family bonding is important for happiness! #Love",
        "Try this delicious pasta recipe! #Foodie",
        "Daily exercise keeps you healthy! #Fitness",
        "This song is a masterpiece! #NowPlaying",
        "Morning workouts boost energy! #Fitness",
        "New skincare routine for glowing skin! #Beauty",
        "Cultural festivals bring people closer! #Culture",
        "The latest smartphones are amazing! #Tech",
        "Best places to visit this summer! #Travel",
        "New designer collections just dropped! #Fashion",
        "Spending time with loved ones is precious! #Family",
        "A balanced diet is key to good health! #Food",
        "Mental health awareness is important! #Health",
        "Listening to music reduces stress! #Music",
        "Yoga is great for mental and physical health! #Fitness",
        "Self-care tips for a better lifestyle! #Beauty"
    ],
    "likes": np.random.randint(50, 1000, 20),
    "retweets": np.random.randint(10, 500, 20)
}

df = pd.DataFrame(data)
print(df.head())


# In[ ]:





# In[6]:


# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print("\nNumber of Duplicate Rows:", duplicate_rows)


# In[7]:


# Get summary statistics for likes & retweets
engagement_stats = df[['likes', 'retweets']].describe()
print(engagement_stats)


# In[8]:


# Group by category and calculate average likes & retweets
category_engagement = df.groupby("category")[["likes", "retweets"]].mean().sort_values(by="likes", ascending=False)

# Display results
print(category_engagement)


# In[9]:


import matplotlib.pyplot as plt

# Plot Likes by Category
plt.figure(figsize=(10, 5))
category_engagement["likes"].plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Average Likes per Category")
plt.xlabel("Category")
plt.ylabel("Average Likes")
plt.xticks(rotation=45)
plt.show()

# Plot Retweets by Category
plt.figure(figsize=(10, 5))
category_engagement["retweets"].plot(kind="bar", color="lightcoral", edgecolor="black")
plt.title("Average Retweets per Category")
plt.xlabel("Category")
plt.ylabel("Average Retweets")
plt.xticks(rotation=45)
plt.show()


# In[10]:


df["text"] = df["text"].str.lower()

import re


df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-zA-Z\s#]", "", str(x)))


# In[11]:


import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")  # Ensure stopwords are downloaded
stop_words = set(stopwords.words("english"))

df["text"] = df["text"].apply(lambda x: " ".join([word for word in str(x).split() if word not in stop_words]))


# In[12]:


from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

df["text"] = df["text"].apply(lambda x: " ".join([word for word in str(x).split() if word not in stop_words]))


# In[13]:


get_ipython().system('pip install --upgrade pillow')


# In[14]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all text data into a single string
all_text = " ".join(df["text"])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Hide axes
plt.show()


# In[15]:


get_ipython().system('pip uninstall -y pillow')


# In[16]:


get_ipython().system('pip install pillow==9.5.0')


# In[18]:


from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER Lexicon
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


# In[19]:


# Function to get sentiment category
def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply to DataFrame
df["Sentiment"] = df["text"].apply(get_sentiment)

# View results
df[["text", "Sentiment"]].head()


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot sentiment counts
plt.figure(figsize=(6,4))
sns.countplot(x=df["Sentiment"], palette="coolwarm")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()


# In[31]:



# Check all column names
print(df.columns)

# Strip spaces and rename correctly
df.rename(columns=lambda x: x.strip(), inplace=True)

# Check again
print(df.columns)


# In[32]:


df.rename(columns={'Sentiment': 'sentiment'}, inplace=True)
print(df.columns)  # Verify if 'sentiment' is now correctly renamed


# In[33]:


print(df["sentiment"].isnull().sum())  # Check for missing values


# In[35]:




# Group sentiment counts by category
sentiment_counts = df.groupby(["category", "sentiment"]).size().unstack()

# Plot grouped bar chart
sentiment_counts.plot(kind="bar", figsize=(10, 6), colormap="coolwarm", edgecolor="black")

# Add labels and title
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Sentiment Distribution Across Categories")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.show()


# In[39]:



from collections import Counter
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Create an engagement score (likes + retweets)
df["engagement"] = df["likes"] + df["retweets"]

# Function to preprocess text (remove stopwords, punctuation, and lowercase)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return words  # Return list instead of a string

# Apply text preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# Store top words per category
category_words = []

for category, group in df.groupby("category"):
    words = [word for text in group["clean_text"] for word in text]  # Flatten list
    word_freq = Counter(words)  # Count word occurrences

    # Weight by engagement
    for word in word_freq:
        word_freq[word] *= group["engagement"].sum()  # Multiply by total engagement

    # Get top 10 words
    top_words = word_freq.most_common(10)

    # Store as a structured list
    for word, score in top_words:
        category_words.append([category, word, score])

# Convert to DataFrame
top_words_df = pd.DataFrame(category_words, columns=["Category", "Word", "Score"])
print(top_words_df)


# In[42]:



from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate word clouds for each category
categories = top_words_df["Category"].unique()
fig, axes = plt.subplots(len(categories), 1, figsize=(10, 5 * len(categories)))

for i, category in enumerate(categories):
    words = top_words_df[top_words_df["Category"] == category]
    word_freq = dict(zip(words["Word"], words["Score"]))
    
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="Set2").generate_from_frequencies(word_freq)
    
    axes[i].imshow(wordcloud, interpolation="bilinear")
    axes[i].set_title(f"Word Cloud for {category}", fontsize=14)
    axes[i].axis("off")

plt.tight_layout()
plt.show()


# In[ ]:




