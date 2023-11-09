import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
file_path = "./data/snopes_test.jsonl"



# Define a function to count words in a given text
def count_words(text):
    words = text.split()
    return len(words)


word_counts = []  # To store word counts
label_list = []

with open(file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        item = json.loads(line)
        text = item.get("txt", "")  # Extract the "txt" value, if not exist return ""
        label = item.get("label", "")
        word_count = count_words(text)
        label_list.append(label)
        word_counts.append(word_count)

# Create a histogram of word counts with intervals of 500 words
plt.figure(figsize=(18, 6))  # Increase the figure size
plt.hist(word_counts, bins=range(0, 5000 + 500, 500), edgecolor='k') #max(word_counts) + 500
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.title('Word Count Histogram')


true_count = 0
false_count = 0

for label in label_list:
    if label == True:
        true_count += 1
    elif label == False:
        false_count += 1

print("Ture: "+ str(true_count) + " False:" + str(false_count))

data_points = []
for i in range(0,5000 + 500, 500):  #max(word_counts) + 500
    interval_count = sum(1 for count in word_counts if i <= count < i + 500)
    data_points.append(interval_count)

# Display the data points count as text above each bar
for i, count in enumerate(data_points):
    if count != 0:
        plt.text(i * 500, count, str(count), ha='center', va='bottom')

plt.grid(True)
plt.savefig("word_count_histogram.png")  # Save the plot to a file


