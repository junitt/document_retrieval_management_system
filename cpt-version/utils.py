from transformers import  BertTokenizer
import torch
import numpy as np
from rouge_chinese import Rouge
from torch.utils.data import DataLoader
import torch
from sklearn import svm
from transformers import BartForConditionalGeneration,AdamW
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import copy
import pandas as pd
import pyarrow.parquet as pq
from math import ceil
def read_article(file_path):
    lines=""
    input_doc=""
    for line in open(file_path,'r',encoding="utf-8"):
        lines+=line
    for sent in lines.split("\n"):
        input_doc+=sent
    return input_doc

def get_doc_clust(clust_doc_path):
    clust_doc=[]
    with open(clust_doc_path, 'r',encoding="utf-8") as f:
        for line in f.readlines():
            if line[-1]=='\n':
                line=line[:-1]
            clust_doc.append(read_article(line))
    return clust_doc

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
        num_beams=6,#这里改过
        length_penalty=1.0,
        max_length=tok_len+2, 
        min_length=sum_min_len,
        no_repeat_ngram_size=2,
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
    lst_ref=tokenizer(ref,add_special_tokens=False)['input_ids']
    n=len(lst_can)
    m=len(lst_ref)
    x=mylcs(lst_can,lst_ref)
    r=x*1.0/m
    p=x*1.0/n
    if x==0:
        return x
    return 2*r*p/(r+p)

rouge=Rouge()
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]

    # # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    ret=""
    for pred in preds:
        if pred != '':
            ret+=pred

    return ret, labels

def compute_metrics(candidate,reference):
    scores = rouge.get_scores(candidate, reference,avg=True)
    for key in scores:
        scores[key]=scores[key]['f']*100

    result=scores

    return result
class SentRep:
    def __init__(self, abs_sent_indices, rel_sent_indices_0_to_10, sent_lens,
        sent_representations_separate, cluster_rep_sent_separate, dist_separate):
        self.abs_sent_indices = abs_sent_indices #(b)
        self.rel_sent_indices_0_to_10 = rel_sent_indices_0_to_10 #(b)
        self.sent_lens = sent_lens #(a)
        self.sent_representations_separate = sent_representations_separate #(c)
        self.cluster_rep_sent_separate = cluster_rep_sent_separate #(c)
        self.dist_separate = dist_separate  #(d)
        self.y = None#rouge-l 分数
        self.binary_y = None

class sentrep_to_nparray:
    def __init__(self):
        self.features_list=self.get_features_list()

    def get_features_list(self):
        features = []
        features.append('abs_sent_indices')
        features.append('rel_sent_indices_0_to_10')
        features.append('sent_lens')
        features.append('sent_representations_separate')
        features.append('cluster_rep_sent_separate')
        features.append('dist_separate')
        features.append('y')
        return features

    def features_to_array(self,sent_reps:SentRep):
        features_list=self.features_list
        x = []
        for feature in features_list:
            val = getattr(sent_reps, feature)
            if is_list_type(val):
                x.extend(val)
            else:
                x.append(val)
        return np.array(x,dtype=np.float32)

def run_training(x, y):
    print("Starting SVR training")
    clf = svm.SVR()
    clf.fit(x, y)
    return clf

def is_end_tok(ch):
    END_TOKENS = ["。", "！", "？", "...", "\u2019", "\u2019"]
    for word in END_TOKENS:
        if(ch==word):
            return True
    return False

def artical_to_sent_list(raw_article:str):
    raw_article=raw_article.replace("\"","")
    raw_article=raw_article.replace("\'","")
    raw_article=raw_article.replace("`","")
    last=0
    ret=[]
    i=0
    while(i<len(raw_article)):
        if(is_end_tok(raw_article[i])):
            if(raw_article[i]=='.'):
                j=i
                while(j<len(raw_article) and raw_article[j]=='.'):
                    j+=1
                ret.append(raw_article[last:j])
                i=j-1
            else:
                ret.append(raw_article[last:i+1])
            last=i+1
        elif i==len(raw_article)-1:
            ret.append(raw_article[last:])
        i+=1
    return ret

def is_special_token(tokenizer,candidate):
    lst=tokenizer.all_special_ids
    for id in lst:
        if id==candidate:
            return True
    return False

def is_list_type(obj):
    return isinstance(obj, (list, tuple, np.ndarray))

#def get_sentrep_id():
    #给定原文，序号以及句子列表，tokenizer，回sentrep

class story_rep_transformer:
    def __init__(self,tokenizer,model,sentence_batch_size,device):
        self.model=model
        self.tokenizer=tokenizer
        self.batch_size=sentence_batch_size
        self.device=device

    def get_sentrep(self,story):
        tokenizer=self.tokenizer
        sent_list=artical_to_sent_list(story)
        sent_tok = tokenizer.batch_encode_plus(sent_list,add_special_tokens =False)
        abs_sent_indices_list=[]
        sent_lens=[]
        sent_representations_separate_lst=[]
        cur_idx=0
        sum_len=0
        for i in range(len(sent_list)):
            abs_sent_indices_list.append(cur_idx)
            sent_lens.append(len(sent_tok['input_ids'][i]))
            cur_idx+=len(sent_tok['input_ids'][i])
        sum_len=cur_idx
        rel_sent_indices_0_to_10=[num*10//sum_len for num in abs_sent_indices_list]
        dat=[]
        cnt=0
        for i in range(ceil(len(sent_list)/self.batch_size)//1):
            temp_lst=[]
            for _ in range(self.batch_size):
                if(cnt==len(sent_list)):
                    break
                temp_lst.append(sent_list[cnt])
                cnt+=1
            dat.append(tuple(temp_lst))
            
        for sent in dat:
            batch_logit=self.get_rep(sent)
            for i in range(batch_logit.shape[0]):
                logit=batch_logit[i].cpu().numpy()
                sent_representations_separate_lst.append((np.mean(logit,axis=0)).reshape(-1))
        cluster_rep_sent_separate=np.mean(np.array(sent_representations_separate_lst),axis=0)
        ret_rep=[]
        for i in range(len(sent_list)):
            ret_rep.append(SentRep(abs_sent_indices_list[i],rel_sent_indices_0_to_10[i],
                                sent_lens[i],sent_representations_separate_lst[i],cluster_rep_sent_separate,
                                cosine_similarity(sent_representations_separate_lst[i].reshape(1, -1),cluster_rep_sent_separate.reshape(1, -1))))
        return ret_rep,sent_list

    def get_rep(self,sent):
        device=self.device
        dct = self.tokenizer.batch_encode_plus(sent, max_length=50, return_tensors="pt",padding='max_length',truncation=True)
            #ans = tokenizer.batch_encode_plus([""], max_length=2, return_tensors="pt", pad_to_max_length=True)
        with torch.no_grad():
            output=self.model(
                    input_ids=dct['input_ids'].to(device),
                    attention_mask=dct['attention_mask'].to(device),
                    output_hidden_states=True,
                    return_dict =True
                )
            return output[3]

class Sim_getter:
    def __init__(self,tokenizer,model,clf:svm.SVR,device):
        self.model=model
        self.tokenizer=tokenizer
        self.clf=clf
        self.rep2nparray=sentrep_to_nparray()
        self.device=device

    def get_rep(self,sent):
        device=self.device
        dct = self.tokenizer.batch_encode_plus(sent, max_length=50, return_tensors="pt",padding='max_length',truncation=True)
            #ans = tokenizer.batch_encode_plus([""], max_length=2, return_tensors="pt", pad_to_max_length=True)
        with torch.no_grad():
            output=self.model(
                    input_ids=dct['input_ids'].to(device),
                    attention_mask=dct['attention_mask'].to(device),
                    output_hidden_states=True,
                    return_dict =True
                )
        return output[3].cpu()
    

    def sim(self,rep:SentRep,summary:str):
        dat=[(summary)]
        copy_rep=copy.deepcopy(rep)
        for sent in dat:
            batch_logit=self.get_rep(sent)
            for i in range(batch_logit.shape[0]):
                logit=batch_logit[i].cpu().numpy()
                copy_rep.cluster_rep_sent_separate=np.mean(logit,axis=0).reshape(-1)
        copy_rep.dist_separate=cosine_similarity(copy_rep.sent_representations_separate.reshape(1, -1),copy_rep.cluster_rep_sent_separate.reshape(1, -1))
        copy_rep.y=0
        return self.predict(copy_rep)
    
    def predict(self,rep:SentRep):
        npmat=self.rep2nparray.features_to_array(rep)
        x=npmat[:-1]
        return self.clf.predict(x.reshape(1, -1))[0]

def choose_correct_token_id(tokenizer,logit):
    lst=np.argsort(-logit[0][-1],axis=0)
    ans=0
    for id in lst:
            if is_special_token(tokenizer,id):
                if id==tokenizer.eos_token_id:
                    ans=id
                    break
            else: 
                ans=id
                break
    return ans

def get_next_word(tokenizer,model:BartForConditionalGeneration,input_seq_list:list,output_seq,max_input_len,max_output_len,device):
    fin_tok=[102]
    for seq_list in input_seq_list:
        seq=""
        list_dct=[]
        for ad_seq in seq_list:
            seq+=ad_seq
        temp_dct = tokenizer.batch_encode_plus([seq],add_special_tokens =False)
        list_dct.extend(temp_dct['input_ids'][0])
        #list_dct.append(tokenizer.sep_token_id)
        fin_tok.extend(list_dct)
    if len(fin_tok)>max_input_len:
        fin_tok=fin_tok[:max_input_len]
    fin_tok[-1]=tokenizer.eos_token_id
    dct={'input_ids':torch.tensor([fin_tok]),'attention_mask':torch.tensor([[1 for i in range(len(fin_tok))]])}
    ans = tokenizer.batch_encode_plus([tokenizer.sep_token+tokenizer.bos_token+output_seq],return_tensors="pt",add_special_tokens =False)
    y_ids = ans['input_ids']
    dec_input_len=len(y_ids[0])
    summaries = model.generate(
        input_ids=dct["input_ids"].to(device),
        attention_mask=dct["attention_mask"].to(device),
        num_beams=1,#这里改过
        early_stopping=True,
        decoder_input_ids=y_ids.to(device),
        max_length=dec_input_len+2, 
        min_length=dec_input_len,
    )   # change these arguments if you want
    dec=0
    token_id=0
    if len(summaries[0])==dec_input_len+1:
        token_id=summaries[0][-1]
        dec = [tokenizer.decode(g[-1], skip_special_tokens=False, clean_up_tokenization_spaces=False) for g in summaries]
    else:
        token_id=summaries[0][-2]
        dec = [tokenizer.decode(g[-2], skip_special_tokens=False, clean_up_tokenization_spaces=False) for g in summaries]
    return (dec[0],token_id)

import pickle
import os 
lamda=0.6
K=5

def mul_predict(tokenizer,model,clust_doc_path,svr_dir,device,L_max=200):
    svr_root=os.path.join(svr_dir,'svr.pickle')
    with open(svr_root, 'rb') as f:
        clf=pickle.load(f)
    clust_doc=get_doc_clust(clust_doc_path)
    s2repmachine=story_rep_transformer(tokenizer,model,sentence_batch_size=8,device=device)
    sim=Sim_getter(tokenizer,model,clf,device=device)
    doc_sent_list=[]
    doc_sent_rep_list=[]
    sent_id=[]
    id_list={}
    row=0
    for doc in clust_doc:
        sent_reps,sent_list=s2repmachine.get_sentrep(doc)
        doc_sent_list.append(sent_list)
        doc_sent_rep_list.append(sent_reps)
        for i in range(len(sent_list)):
            sent_id.append((row,i))
            id_list[(row,i)]=len(sent_id)-1
        row+=1#下一个文档
    doc_sent_I_list=[]#储藏I的值
    doc_sent_mmr_list=[]
    for sent_reps in doc_sent_rep_list:
        doc_i_list=[sim.predict(rep) for rep in sent_reps]
        doc_sent_I_list.append(doc_i_list)
        doc_sent_mmr_list.extend([lamda*vi for vi in doc_i_list])
    summary=""
    sum_list=[]
    for _ in range(L_max):
        npa=np.argsort(np.array(doc_sent_mmr_list))
        topk=npa[max(0,len(npa)-K):]
        tuple_list=[sent_id[i] for i in topk]
        tuple_list=sorted(tuple_list)
        input_seq_list=[]
        temp=[]
        for i in range(min(len(tuple_list),K)):
            v=tuple_list[i]
            if i==0 or tuple_list[i][0]==tuple_list[i-1][0]:
                temp.append(doc_sent_list[v[0]][v[1]])
            else:
                input_seq_list.append(temp)
                temp=[doc_sent_list[v[0]][v[1]]]
        input_seq_list.append(temp)
        word,token_id=get_next_word(tokenizer,model,input_seq_list,summary,1000,L_max,device=device)
        if is_special_token(tokenizer,token_id):
            break
        sum_list.append(word)
        summary+=word
        for i in range(len(doc_sent_I_list)):
            v=sent_id[i]
            doc_sent_mmr_list[i]=lamda*doc_sent_I_list[v[0]][v[1]]-(1-lamda)*sim.sim(doc_sent_rep_list[v[0]][v[1]],summary)
    return summary
