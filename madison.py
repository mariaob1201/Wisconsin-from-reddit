import os
from dotenv import load_dotenv
import praw
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import pandas as pd
from datetime import datetime

# Function to get sentiment polarity
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # Range from -1 (negative) to 1 (positive)

# Load environment variables
load_dotenv()

# Configure Reddit API
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('USER_AGENT')
)

# Search query and subreddit
query = "Madison"
subreddit = reddit.subreddit("all")
search_limit = 50

# Create output directory if it doesn't exist
output_dir = 'wisconsin'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Store submissions to avoid multiple API calls
submissions_list = list(subreddit.search(query, sort="new", limit=search_limit))

# Print debug information
print(f"Number of submissions found: {len(submissions_list)}")

# Extract text content for word cloud
text_content = ""
for submission in submissions_list:
    text_content += submission.title + " " + submission.selftext + " "

# Print the length of text content
print(f"Total text content length: {len(text_content)} characters")

# Check if we have content before creating word cloud
if len(text_content.strip()) > 0:
    # Generate the Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          colormap='viridis', min_word_length=3).generate(text_content)

    # Display the Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud - Wisconsin on Reddit", fontsize=14)
    plt.savefig(os.path.join(output_dir, 'cloud_Wisconsin.png'))
    plt.show()
else:
    print("Not enough text content to generate a word cloud.")
    # If we need a fallback word cloud for testing
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate("No results found")

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("No Results Found", fontsize=14)
    plt.savefig(os.path.join(output_dir, 'cloud2_Wisconsin.png'))
    plt.show()

# Process text only if we have content
if text_content.strip():
    # Download stopwords if you don't have them
    nltk.download('stopwords', quiet=True)

    # Filter stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in text_content.split() if word.lower() not in stop_words and len(word) > 1]

    print(f"Number of words after filtering: {len(filtered_words)}")

    if filtered_words:
        # Count the most common words
        word_counts = Counter(filtered_words)

        # Show the 10 most common words (or fewer if we have less than 10)
        max_words = min(10, len(word_counts))
        common_words = word_counts.most_common(max_words)
        print(f"Top {max_words} common words:", common_words)
    else:
        print("No words remained after filtering.")
else:
    print("No text content to analyze for word frequency.")

# Analyze sentiment of posts only if there are submissions
sentiment_scores = []
post_data = []

if submissions_list:
    for submission in submissions_list:
        # Sentiment analysis
        combined_text = submission.title + " " + submission.selftext
        sentiment = get_sentiment(combined_text) if combined_text.strip() else 0
        sentiment_scores.append(sentiment)

        # Post metadata
        post_data.append({
            'title': submission.title,
            'comments': submission.num_comments,
            'upvotes': submission.score,
            'sentiment': sentiment,
            'date': datetime.fromtimestamp(submission.created_utc)  # Convert to datetime
        })

    # Calculate average sentiment
    if sentiment_scores:
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        print(f"\nAverage sentiment for 'Wisconsin': {average_sentiment}")
    else:
        print("\nNo text content available for sentiment analysis")

    # Post statistics
    print(f"\nPosts about 'Wisconsin' - Number of posts: {len(post_data)}")

    if post_data:
        max_comments_post = max(post_data, key=lambda x: x['comments'])
        print(f"Top post by comments: {max_comments_post['title']} ({max_comments_post['comments']} comments)")

        max_upvotes_post = max(post_data, key=lambda x: x['upvotes'])
        print(f"Top post by upvotes: {max_upvotes_post['title']} ({max_upvotes_post['upvotes']} upvotes)")

    # Date analysis
    if post_data:
        # Create a DataFrame from post_data
        df = pd.DataFrame(post_data)

        # Ensure 'date' column is in datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Extract just the date portion (without time)
        df['date'] = df['date'].dt.date

        # Count posts by date
        posts_per_day = df['date'].value_counts().sort_index()

        # Plot the trend
        plt.figure(figsize=(10, 5))
        posts_per_day.plot(kind='line', title='Posts per Day about Wisconsin')
        plt.xlabel('Date')
        plt.ylabel('Number of Posts')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'posts_per_day_wisconsin.png'))
        plt.show()

else:
    print("No posts found matching the search criteria.")
    print("You may want to try a different search query or check your Reddit API credentials.")
