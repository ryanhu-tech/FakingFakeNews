import os
import xml.etree.ElementTree as ET
import json

# Define the directory containing the XML files
xml_directory = "C:\dataset\TL17\ltf"

# Create a list to store the extracted data
data_list = []

# Iterate through the XML files in the directory
for filename in os.listdir(xml_directory):
    if filename.endswith(".xml"):
        file_path = os.path.join(xml_directory, filename)

        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Find all <ORIGINAL_TEXT> elements within the file
        original_text_elements = root.findall(".//ORIGINAL_TEXT")

        # Combine the text from all <ORIGINAL_TEXT> elements into one string
        combined_text = " ".join([element.text for element in original_text_elements])

        if combined_text:
            # Create a unique identifier based on the filename
            unique_id = filename.replace(".ltf.xml", "")

            # Create a dictionary with the unique ID and combined text
            data_entry = {"id": unique_id, "txt": combined_text}
            data_list.append(data_entry)
        else:
            print("<ORIGINAL_TEXT> not found or is empty in file: {file_path}")

# Save the extracted data in JSONL format
jsonl_file = "TL17.jsonl"
with open(jsonl_file, "w", encoding="utf-8") as jsonl:
    for entry in data_list:
        jsonl.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Data has been extracted and saved to {jsonl_file}")