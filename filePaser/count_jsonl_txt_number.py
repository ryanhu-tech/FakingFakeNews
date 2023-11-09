import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
file_path = "TL17.jsonl"



# Define a function to count words in a given text
def count_words(text):
    words = text.split()
    return len(words)


word_counts = []  # To store word counts

with open(file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        item = json.loads(line)
        text = item.get("text", "")  # Extract the "txt" value, if not exist return ""
        word_count = count_words(text)
        word_counts.append(word_count)

# Create a histogram of word counts with intervals of 500 words
plt.figure(figsize=(18, 6))  # Increase the figure size
plt.hist(word_counts, bins=range(0, 5000 + 500, 500), edgecolor='k') #max(word_counts) + 500
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Word Count Histogram')


true_count = 0
false_count = 0



data_points = []
for i in range(0,5000+ 500, 500):  #max(word_counts) + 500
    interval_count = sum(1 for count in word_counts if i <= count < i + 500)
    data_points.append(interval_count)

# Display the data points count as text above each bar
for i, count in enumerate(data_points):
    if count != 0:
        plt.text(i * 500, count, str(count), ha='center', va='bottom')

plt.grid(True)
plt.savefig("TL17_count_histogram.png")  # Save the plot to a file


