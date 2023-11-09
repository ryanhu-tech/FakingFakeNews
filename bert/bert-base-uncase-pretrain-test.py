from transformers import BertTokenizer, BertForPreTraining
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased')

with open('clean.txt', 'r') as fp:
    text = fp.read().split('\n') #根據\n切割

#Preparing For NSP
bag = [item for sentence in text for item in sentence.split('.') if item != '']  #把全部文章根據句點切割同時存在一個list
bag_size = len(bag)

import random

sentence_a = []
sentence_b = []
label = []

#每次讀取一個文章，再從裡面分割句子出來
for paragraph in text:
    sentences = [
        sentence for sentence in paragraph.split('.') if sentence != ''
    ]  #僅存一篇文章中的句子
    num_sentences = len(sentences)
    if num_sentences > 1:
        start = random.randint(0, num_sentences-2) #如果只有三句，第一句一定是從0開始
        # 50/50 whether is IsNextSentence or NotNextSentence
        if random.random() >= 0.5:
            # this is IsNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(sentences[start+1])
            label.append(0)
        else:
            index = random.randint(0, bag_size-1)  #如果不是Next Sentence，則從總文章分段去抽，但這樣其實有機會中next sentence
            #this is NotNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(bag[index])
            label.append(1)

#句子數量會與setence_a,_b一樣 317
inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

inputs['next_sentence_label'] = torch.LongTensor([label]).T #將label轉成向量, 存在key為'next_sentence_label'中

inputs['labels'] = inputs.input_ids.detach().clone() #the labels tensor is simply a clone of the input_ids tensor before masking，用在MLMS，把每個句子中的每個字的id視為label
print(inputs.keys())

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
#隨機到<0.15而且不是CLS，SEP及PAD，其中的值會等於True，表示要mask的部分
mask_arr =  (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0) # CLS (101), SEP (102), and PAD (0) tokens.

selection = []
for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist() # noonzer() 會把 Ture部分的id回傳
        #這邊mask_arr[i]
    )

for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]]= 103 #表示mask掉id的位置

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = MeditationsDataset(inputs)

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


from transformers import AdamW

model.train()
optim = AdamW(model.parameters(), lr=5e-5)

from tqdm import tqdm

epochs = 2

for epoch in range(epochs):
    loop = tqdm(loader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        next_sentence_label = batch['next_sentence_label'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        next_sentence_label= next_sentence_label, labels=labels)

        loss = outputs.loss
        loss.backward()
        optim.step()
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())