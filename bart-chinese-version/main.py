from transformers import  BartForConditionalGeneration,BertTokenizer
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from lcstsdataset import lcstsdataset
from transformers import AdamW, get_linear_schedule_with_warmup
#from modeling_cpt import BartForConditionalGeneration
from torch.utils.tensorboard import SummaryWriter
import os
lr=2e-5
accumulation_steps=2
local_batch_size=16
sum_maxlength=100
max_length=125
warm_up_ratio=0.1
num_epochs=50
overall_step=0
data_dir='./data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter()
tokenizer = BertTokenizer.from_pretrained("./bart-base-chinese")
model = BartForConditionalGeneration.from_pretrained("./bart-base-chinese").to(device)
no_decay_params = ["bias", "LayerNorm", "layer_norm"]
params = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if (not any(nd in n for nd in no_decay_params)) and ("crf." not in n)
        ],
        "weight_decay": 1e-2,
    },
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay_params)
        ],
        "weight_decay": 0.0,
    },
]
optimizer=AdamW(params=params,lr=lr,eps=1e-8)

def train_epoch(train_dataloader,epoch,scheduler):
    bar = tqdm(train_dataloader)
    bar.set_description(f'epoch {epoch:2}')
    global overall_step
    for batch_id, (stories,refers) in enumerate(bar):
        overall_step+=1
        dct = tokenizer.batch_encode_plus(stories, max_length=max_length, return_tensors="pt",padding='max_length',truncation=True)
        ans = tokenizer.batch_encode_plus(refers, max_length=sum_maxlength, return_tensors="pt",padding='max_length',truncation=True)
        pad_token_id = tokenizer.pad_token_id
        y = ans['input_ids']
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
            
        if type (model).__name__=='BartForConditionalGeneration':
                output = model(
                    input_ids=dct['input_ids'].to(device),
                    attention_mask=dct['attention_mask'].to(device),
                    decoder_input_ids=y_ids.to(device),
                    labels =lm_labels.to(device),
                    #output_hidden_states=True,
                )
        else:
                output = model(
                    input_ids=dct['input_ids'].to(device),
                    attention_mask=dct['attention_mask'].to(device),
                    decoder_input_ids=y_ids.to(device),
                    output_hidden_states =True,
                )
        loss= output[0]
        writer.add_scalar('Loss/train',loss.item(), overall_step)
        bar.set_postfix(loss=loss.item())
        loss = loss/accumulation_steps
        loss.backward()
        if((overall_step+1)%accumulation_steps)==0:
            optimizer.step()        # 反向传播，更新网络参数
            optimizer.zero_grad()   # 清空梯度
            scheduler.step()
        
        
def eval_epoch(train_dataloader,epoch):
    bar = tqdm(train_dataloader)
    bar.set_description(f'test epoch {epoch:2}')
    eval_loss=0
    steps=0
    for batch_id, ( stories, refers) in enumerate(bar):
        steps+=1
        dct = tokenizer.batch_encode_plus(stories, max_length=max_length,return_tensors="pt",padding='max_length',truncation=True)
        ans = tokenizer.batch_encode_plus(refers, max_length=sum_maxlength, return_tensors="pt",padding='max_length',truncation=True)

        # further process
        pad_token_id = tokenizer.pad_token_id
        y = ans['input_ids']
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == pad_token_id] = -100
        with torch.no_grad():
            if type (model).__name__=='BartForConditionalGeneration':
                output = model(
                    input_ids=dct['input_ids'].to(device),
                    attention_mask=dct['attention_mask'].to(device),
                    decoder_input_ids=y_ids.to(device),
                    labels=lm_labels.to(device),
                    output_hidden_states=True
                )
            else :
                output = model(
                    input_ids=dct['input_ids'].to(device),
                    attention_mask=dct['attention_mask'].to(device),
                    decoder_input_ids=y_ids.to(device),
                    encoder_last_hidden_state =True,
                )

        loss = output[0]
        eval_loss += loss.mean().item()
        bar.set_postfix(loss=loss.item())
    fin_loss=eval_loss/steps
    print("eval loss: "+str(fin_loss))
    return fin_loss

def save_model(model_dir):
    global model
    if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    model.save_pretrained(model_dir)

def main():
    global local_batch_size
    train_dataset=lcstsdataset(os.path.join(data_dir,'train.json'))
    train_dataloader=DataLoader(train_dataset, batch_size=local_batch_size, shuffle=True)
    ans=0
    for batch_id, (stories,refers) in enumerate(train_dataloader):
        if len(tokenizer(stories[0])['input_ids'])>120:
            ans+=1
    print(ans)
    t_total=(len(train_dataloader)*num_epochs)//(accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * t_total, num_training_steps=t_total)
    valid_dataset=lcstsdataset(os.path.join(data_dir,'test.json'))
    valid_dataloader=DataLoader(valid_dataset, batch_size=local_batch_size, shuffle=True)

    ans=1000
    save_path='./bart-fine-tuned-chinese'
    for i in range(num_epochs):
        train_epoch(train_dataloader,i+1,scheduler)
        fin_loss=eval_epoch(valid_dataloader,i+1)
        if fin_loss<ans:
            ans=fin_loss
            save_model(save_path)

if __name__=='__main__':
    main()