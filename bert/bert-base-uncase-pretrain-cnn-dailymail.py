from transformers import AutoTokenizer
import multiprocessing
from tqdm import tqdm
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from datasets import load_dataset
# Split the dataset into training, validation, and test sets
train_data = load_dataset("cnn_dailymail", "3.0.0", split="train")
val_data = load_dataset("cnn_dailymail", "3.0.0", split="validation")
test_data = load_dataset("cnn_dailymail", "3.0.0", split="test")


tokenizer = AutoTokenizer.from_pretrained("tokenizer")
num_proc = multiprocessing.cpu_count()
print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

def group_texts(examples):
    tokenized_inputs = tokenizer(
       examples["article"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

# preprocess dataset
tokenized_datasets = train_data.map(group_texts, batched=True, remove_columns=["article"], num_proc=num_proc)
tokenized_datasets.features