import convokit
from convokit import Corpus, download
import random
from langdetect import detect
import csv

# Load the corpus
# Assuming you have a subreddit-Denmark corpus already downloaded
#corpus_path = "path_to_your_corpus"  # Replace with the actual path or use Convokit.download() if hosted
#corpus = Corpus(filename=corpus_path)
corpus = Corpus(download('subreddit-Denmark'))

# Filter posts with more than 100 tokens and a minimum number of upvotes
its = 0
posts_with_100_tokens_and_upvotes = []
min_upvotes = 10  # Set your minimum upvote threshold here
lower_bound, upper_bound = 50, 1000

for conversation in corpus.iter_conversations():
    if its < 10000:
        # Assuming the first utterance in the conversation is the original post
        first_utterance = next(conversation.iter_utterances())
        upvotes = first_utterance.meta.get('score')  # Adjust this if upvotes are stored differently
        if (lower_bound <= len(first_utterance.text.split()) <= upper_bound) and (upvotes >= min_upvotes):
            if detect(first_utterance.text) == "da":
                posts_with_100_tokens_and_upvotes.append((conversation.meta.get('title', 'No Title'), first_utterance.text, upvotes))
                its += 1

print("Amount of samples:", len(posts_with_100_tokens_and_upvotes))

with open('reddit.csv', 'w', newline='', encoding='utf-8') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['title', 'text', 'upvotes'])
    for row in posts_with_100_tokens_and_upvotes:
        csv_out.writerow(row)

"""# Print the filtered posts with their titles and upvotes
for title, text, upvotes in posts_with_100_tokens_and_upvotes:
    print(f"Title: {title}")
    print(f"Post: {text}")
    print(f"Upvotes: {upvotes}\n")"""