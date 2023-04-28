from transformers import  BertTokenizer
import torch
import argparse
from modeling_cpt import CPTForConditionalGeneration
from utils import get_pred,get_vec,get_vec_avg
if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default='/path/to/model',type=str)
    parser.add_argument("--file_path",default="/path/to/dataset/",type=str)
    parser.add_argument("--sum_min_len",default=30,type=int)
    parser.add_argument("--gen_vec",default=0,type=int)
    args = parser.parse_args()
    arg_dict=args.__dict__
    tokenizer = BertTokenizer.from_pretrained(arg_dict['model_path'])
    model = CPTForConditionalGeneration.from_pretrained(arg_dict['model_path']).to(device)
    lines=""
    input_doc=""
    for line in open(arg_dict['file_path'],'r',encoding="utf-8"):
        lines+=line
    for sent in lines.split("\n"):
        input_doc+=sent
    s=get_pred(tokenizer,model,input_doc,sum_min_len=int(arg_dict['sum_min_len']),device=device)
    #rep=get_vec(device,model,tokenizer,input_doc,max_length=1000)
    rep=get_vec_avg(device,model,tokenizer,input_doc,max_length=1000)
    if arg_dict['gen_vec']>0:
        print(rep.tolist())
    else:
        print(s)