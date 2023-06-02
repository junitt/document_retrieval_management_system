from transformers import  BertTokenizer
from modeling_cpt import CPTForConditionalGeneration
import torch
import argparse
from utils import get_pred,get_vec,get_vec_avg,read_article,mul_predict
if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default='/path/to/model',type=str)
    parser.add_argument("--file_path",default="/path/to/dataset/",type=str)
    parser.add_argument("--file__type",default=0,type=int)
    parser.add_argument("--gen_vec",default=0,type=int)
    parser.add_argument("--sum_min_len",default=10,type=int)
    parser.add_argument("--svr_dir",default=None,type=str)#新加
    args = parser.parse_args()
    arg_dict=args.__dict__
    tokenizer = BertTokenizer.from_pretrained(arg_dict['model_path'])
    model = CPTForConditionalGeneration.from_pretrained(arg_dict['model_path']).to(device)
    if arg_dict['file__type']==0:
        input_doc=read_article(arg_dict['file_path'])
        s=get_pred(tokenizer,model,input_doc,sum_min_len=int(arg_dict['sum_min_len']),device=device)
        #rep=get_vec(device,model,tokenizer,input_doc,max_length=1000)
        rep=get_vec_avg(device,model,tokenizer,input_doc,max_length=1000)
        if arg_dict['gen_vec']>0:
            print(rep.tolist())
        else:
            print(s)
    else :
        summary=mul_predict(tokenizer,model,arg_dict['file_path'],arg_dict['svr_dir'],device)
        print(summary)