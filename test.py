from transformers import BartForConditionalGeneration, BertTokenizer
import torch
from utils import calc_rouge_l,get_pred,compute_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm
from lcstsdataset import lcstsdataset
from modeling_cpt import CPTForConditionalGeneration
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained("./model")
model = CPTForConditionalGeneration.from_pretrained("./model").to(device)
max_length=120
data_dir='./data'
def get_pred_and_ref(stories,summary_lines):
    dct = tokenizer.batch_encode_plus(stories, max_length=max_length, return_tensors="pt",padding='max_length',truncation=True)
    global model 
    summaries = model.generate(
        input_ids=dct["input_ids"].to(device),
        attention_mask=dct["attention_mask"].to(device),
        num_beams=6,
        length_penalty=1.0,
        max_length=50,  # +2 from original because we start at step=1 and stop before max_length
        min_length=8,  # +1 from original because we start at step=1
        no_repeat_ngram_size=2,
        do_sample=True,
    )  # change these arguments if you want

    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
    return {'pred':dec,'reference':summary_lines}

ans=0
total=0

def calc_score():
    global total
    global ans
    test_dataset=lcstsdataset(os.path.join(data_dir,'test.json'))
    test_dataloader=DataLoader(test_dataset, batch_size=2, shuffle=False)
    for ( stories, refers) in tqdm(test_dataloader):
        rec=get_pred_and_ref(stories, refers)
        for i in range(len(rec['pred'])):
            vald=calc_rouge_l(tokenizer=tokenizer,candidate=rec['pred'][i],ref=rec['reference'][i])
            #vald=compute_metrics(decoded_preds=rec['pred'][i],decoded_labels=rec['reference'][i])
            total+=1
            ans+=vald
    print(ans*1.0/total)

if __name__=='__main__':
    calc_score()