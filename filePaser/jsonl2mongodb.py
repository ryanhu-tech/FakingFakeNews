import json
from pymongo import MongoClient

# MongoDB connection settings
client = MongoClient("mongodb://localhost:27017")
db = client['Propaganda_Seed']  # Replace with your database name
collection = db['TL17']  # Replace with your collection name

# Path to your JSONL file
jsonl_file = "TL17.jsonl"  # Replace with your file name

# Open the JSONL file and insert data into MongoDB
with open(jsonl_file, "r") as file:
    for line in file:
        data = json.loads(line)
        collection.insert_one(data)

# Close the MongoDB connection
client.close()