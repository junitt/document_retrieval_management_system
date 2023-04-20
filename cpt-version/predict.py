from transformers import  BertTokenizer
import torch
import argparse
from modeling_cpt import CPTForConditionalGeneration
from utils import get_pred
if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",default='/path/to/model',type=str)
    parser.add_argument("--file_path",default="/path/to/file/",type=str)
    parser.add_argument("--sum_min_len",default=20,type=int)
    args = parser.parse_args()
    arg_dict=args.__dict__
    tokenizer = BertTokenizer.from_pretrained(arg_dict['model_path'])
    model = CPTForConditionalGeneration.from_pretrained(arg_dict['model_path']).to(device)
    lines=""
    input_doc=""
    for line in open(arg_dict['file_path'],'r'):
        lines+=line
    for sent in lines.split("\n"):
        input_doc+=sent
    s=get_pred(tokenizer,model,input_doc,sum_min_len=int(arg_dict['sum_min_len']))
    print(s)