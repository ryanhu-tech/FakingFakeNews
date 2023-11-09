from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import random
import time
import os
from sklearn.metrics import f1_score, roc_auc_score

seed = 42
# fix random seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = True

# define model

class BERTModelForClassification(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        hidden_states = self.bert(input_ids=input_ids,
                    attention_mask=attention_mask)[0] # batch, seq_len, emb_dim

        # get [CLS] embeddings
        """這一行程式碼是從 hidden_states 中選取每個序列（句子）的第一個標記，通常是 [CLS] 標記，
        以獲得對整個句子的表示。cls_embeddings 是一個包含每個句子的 [CLS] 表示的張量。"""
        cls_embeddings = hidden_states[:,0,:]
        logits = self.linear(cls_embeddings)
        #這段程式碼表示了一個簡單的文本分類模型，它使用 BERT 模型來提取文本的語義信息，然後通過線性層對其進行分類，最終得到文本的預測概率。
        outputs = torch.sigmoid(logits)
        return outputs        


# define loader
class PropaFakeDataset(Dataset):
    def __init__(self, jsonl_path):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.data = []
        
        for line in open(jsonl_path,'r'): #打開指定路徑的所有 JSONL 檔案
            inst = json.loads(line) #對每一行，使用 json.loads() 函數將 JSON 格式的文本轉換為 Python 的字典物件
            label = inst['label'] #獲取 JSON 對象中的 'label' 欄位的值
            inputs = self.tokenizer(inst['txt'], max_length=args.max_sequence_length, padding="max_length", truncation=True)

            """
            attention_mask 只有0或1，與文本中的每個標記（token）相對應。這個序列指示了哪些標記應該受到模型的關注，哪些應該被忽略。比如說多序列句子下，會有長有短，較短的部分會在後面補0。
            此時需要把attention_mask後面沒對應到的文字也設為0。
            """

            self.data.append({     #處理後的資料以字典的形式添加到 self.data 列表中
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'label': label
            })
            
        
    def __len__(self):
        # 200K datapoints
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]['input_ids'], self.data[idx]['attention_mask'], self.data[idx]['label']
    
    def collate_fn(self, batch): #這個函數的主要目的是將這個批次的數據組織成 PyTorch 張量，以便進行後續的模型訓練。
        # print(batch)
        input_ids = torch.cuda.LongTensor([inst[0] for inst in batch])
        attention_masks = torch.cuda.LongTensor([inst[1]for inst in batch])
        labels = torch.cuda.FloatTensor([inst[2]for inst in batch])

        return input_ids, attention_masks, labels

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('--max_sequence_length', default=512, type=int)
parser.add_argument('--model_name', default='facebook/bart-large') #原始roberta-large
#parser.add_argument('--checkpoint_path', default='../bert2bert_cnn_daily_mail/pytorch_model.bin', type=str, required=False)
parser.add_argument('--data_dir', default='../data/')
parser.add_argument('--warmup_epoch', default=5, type=int) #原本是5，主要目的是在訓練的早期階段，讓學習率保持較小的值，然後逐漸增加，以幫助模型更穩定地收斂到合適的權重
parser.add_argument('--max_epoch', default=30, type=int)#原本是30
parser.add_argument('--batch_size', default=2, type=int)#原本是2
parser.add_argument('--eval_batch_size', default=2, type=int)
parser.add_argument('--accumulate_step', default=8, type=int) #原本是8 用於控制梯度累積的參數，主要用於處理內存問題、提高訓練穩定性以及控制訓練時間
parser.add_argument('--output_dir', default='../output/', required=False) #雖然過程不會存在這，但沒設定default會出錯

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

output_dir = os.path.join(args.output_dir, timestamp)
os.makedirs(output_dir)
# init model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = BERTModelForClassification(args.model_name).cuda()
#print(model.state_dict())

#這邊是要放入intermediate pre-trained model
#model_path = args.checkpoint_path

#checkpoint = torch.load(model_path)

#print(checkpoint.keys())


#model.load_state_dict(checkpoint['model'], strict=True) #要記起來為何這邊會出錯
#model.load_state_dict(checkpoint, strict=True) #要記起來為何這邊會出錯




# init loader
train_set = PropaFakeDataset(os.path.join(args.data_dir,'train.jsonl')) #要先產生PropaFakeDataset類別，才能放入DataLoader
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
dev_set = PropaFakeDataset(os.path.join(args.data_dir,'dev.jsonl'))
dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn)
test_set = PropaFakeDataset(os.path.join(args.data_dir,'test.jsonl'))
test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn)

# define loss
critera = nn.BCELoss()

state = dict(model=model.state_dict()) #model.state_dict()：用於獲取模型的狀態字典，state 的變數將一個字典賦值給它。這個字典中只有一個鍵值對，鍵是 'model'，值是模型的狀態字典

# optimizer
param_groups = [
    {
        'params': [p for n, p in model.named_parameters() if n.startswith('bert')], #創建一個包含模型中所有以 'bert' 開頭的參數的列表
        'lr': 5e-5, 'weight_decay': 1e-05
    },
    {
        'params': [p for n, p in model.named_parameters() if not n.startswith('bert')],
        'lr': 1e-3, 'weight_decay': 0.001
    },
    
]

batch_num = len(train_set) // (args.batch_size * args.accumulate_step) #batch_num最終是要考慮 bathc_size * accumulate_step，代表真正更新權重的batch數
+ (len(train_set) % (args.batch_size * args.accumulate_step) != 0)

optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=batch_num*args.warmup_epoch, #這些step 都是要考慮accumulate_step
                                           num_training_steps=batch_num*args.max_epoch)

best_dev_accuracy = 0
model_path = os.path.join(output_dir,'best.pt')
for epoch in range(args.max_epoch):
    training_loss = 0
    model.train()
    for batch_idx, (input_ids, attn_mask, labels) in enumerate(tqdm(train_loader)):        
        

        
        outputs = model(input_ids, attention_mask=attn_mask).view(-1)

        # loss
        
        loss = critera(outputs, labels)
        loss.backward()
        training_loss += loss.item()
        if (batch_idx + 1) % args.accumulate_step == 0:
            torch.nn.utils.clip_grad_norm_(  #對模型的梯度進行裁剪。model.parameters() 表示獲取模型的所有參數（權重和梯度），而 5.0 是裁剪的閾值，表示梯度的範數不應該超過 5.0
                model.parameters(), 5.0)
        
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

    print(f"Trainin Loss: {training_loss:4f}" )
    # train the last batch
    if batch_num % args.accumulate_step != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()

    # validation
    with torch.no_grad():
        model.eval()
        dev_outputs = []
        dev_labels = []
        for _, (input_ids, attn_mask, labels) in enumerate(dev_loader):
            outputs = model(input_ids, attention_mask=attn_mask).view(-1)
            dev_outputs.append(outputs) 
            dev_labels.append(labels)
        dev_outputs = torch.cat(dev_outputs, dim=0) # n_sample,
        dev_labels = torch.cat(dev_labels, dim=0) # n_sample,
        
        dev_outputs_bool = dev_outputs > 0.5    #值大於0.5就為 true
        dev_labels_bool = dev_labels == 1 # convert to float tensor
        

        dev_accuracy = torch.sum(dev_labels_bool == dev_outputs_bool) / len(dev_labels)
        
        dev_auc = roc_auc_score(dev_labels.cpu().numpy(), dev_outputs.detach().cpu().numpy())
        print(f"Dev AUC: {dev_auc}. ")
        
        dev_f1 = f1_score(dev_labels.cpu().numpy(), np.array([1 if l > 0.5 else 0 for l in dev_outputs]))
        print(f"Dev F1: {dev_f1}. ")
        
        dev_accuracy = dev_auc

        if dev_accuracy > best_dev_accuracy:
            model_path = os.path.join(output_dir, 'Epoch_'+str(epoch) +'_best.pt')
            print(f"Saving to {model_path}")
            best_dev_accuracy = dev_accuracy
            torch.save(state, model_path)

        print(f"Epoch {epoch} dev accuracy: {dev_accuracy * 100:.2f}. Best dev accuracy: {best_dev_accuracy*100:.2f}.")            


checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'], strict=True)    
test_output_file = os.path.join(output_dir, 'test_pred.json')

with torch.no_grad():
    model.eval()
    test_outputs = []
    test_labels = []
    
    for _, (input_ids, attn_mask, labels) in enumerate(test_loader):
        outputs = model(input_ids, attention_mask=attn_mask).view(-1)
        test_outputs.append(outputs) 
        test_labels.append(labels)
    test_outputs = torch.cat(test_outputs, dim=0) # n_sample,
    test_labels = torch.cat(test_labels, dim=0) # n_sample,
    
    test_outputs_bool = test_outputs > 0.5
    test_labels_bool = test_labels == 1 # convert to float tensor

    test_accuracy = torch.sum(test_outputs_bool == test_labels_bool) / len(test_labels)
    print(f"Epoch {epoch} test accuracy: {test_accuracy*100:.2f}. ")
    
    test_auc = roc_auc_score(test_labels.cpu().numpy(), test_outputs.detach().cpu().numpy())
    print(f"Test AUC: {test_auc}. ")
    
    test_f1 = f1_score(test_labels.cpu().numpy(), np.array([1 if l > 0.5 else 0 for l in test_outputs]))
    print(f"Test F1: {test_f1}. ")
    
    test_outputs = [float(o) for o in test_outputs]
       
    with open(test_output_file,'w') as f:
        json.dump({'output':test_outputs}, f)
print("model path:", model_path)