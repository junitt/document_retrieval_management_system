from transformers import  BertTokenizer
import torch
import numpy as np
def mylcs(lst1:list,lst2:list):
    ans=0
    n=len(lst1)
    m=len(lst2)
    lst1.insert(0,0)
    lst2.insert(0,0)
    dp=[[0 for _ in range(m+1)]for i in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            if lst1[i]==lst2[j]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
            ans=max(ans,dp[i][j])
    return ans

def get_pred(tokenizer,model,story,sum_min_len=8,device='cuda'):
    tok_len=min(1000,len(tokenizer(story)["input_ids"])+5)
    dct = tokenizer.batch_encode_plus([story], max_length=tok_len,return_tensors="pt",padding='max_length',truncation=True)
    summaries = model.generate(
        input_ids=dct["input_ids"].to(device),
        attention_mask=dct["attention_mask"].to(device),
        num_beams=4,
        length_penalty=1.0,
        max_length=tok_len+2, 
        min_length=sum_min_len,
        no_repeat_ngram_size=3,
        do_sample=True,
    )  # change these arguments if you want

    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
    return dec

def get_mat(device,model,tokenizer,article,max_length=1000):
    dct = tokenizer.batch_encode_plus([article], max_length=1000, return_tensors="pt",padding='max_length',truncation=True)
    with torch.no_grad():
            output=model(
                    input_ids=dct['input_ids'].to(device),
                    attention_mask=dct['attention_mask'].to(device),
                    output_hidden_states=True,
                    return_dict =True
                )
            return output[3][0].cpu()

def get_vec(device,model,tokenizer,article,max_length=1000):
    mat=get_mat(device,model,tokenizer,article,max_length)
    return np.array(mat).reshape(-1,)

def get_vec_avg(device,model,tokenizer,article,max_length=1000):
    mat=get_mat(device,model,tokenizer,article,max_length)
    arr=np.mean(np.array(mat),axis=0)
    return arr.reshape(-1,)

def calc_rouge_l(tokenizer:BertTokenizer,candidate:str,ref:str):
    lst_can=tokenizer(candidate,add_special_tokens=False)['input_ids']
    lst_ref=tokenizer(ref)['input_ids']
    n=len(lst_can)
    m=len(lst_ref)
    x=mylcs(lst_can,lst_ref)
    r=x*1.0/m
    p=x*1.0/n
    if x==0:
        return x
    return 2*r*p/(r+p)
#中 国 零 售 企 业 营 业 收 入 最 高
#家电专业店营收增长净利润下滑