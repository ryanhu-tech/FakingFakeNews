#https://huggingface.co/blog/pretraining-bert

from tqdm import tqdm
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset
# Split the dataset into training, validation, and test sets
train_data = load_dataset("cnn_dailymail", "3.0.0", split="train")
val_data = load_dataset("cnn_dailymail", "3.0.0", split="validation")
test_data = load_dataset("cnn_dailymail", "3.0.0", split="test")


# repositor id for saving the tokenizer
tokenizer_id="bert-base-uncased-cnn-dailymaiil"

# create a python generator to dynamically load the data
def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, len(train_data), batch_size)):
        yield train_data[i : i + batch_size]["article"]

# create a tokenizer from existing one to re-use special tokens
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
bert_tokenizer.save_pretrained("tokenizer")

