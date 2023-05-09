from torch.utils.data import Dataset
import json
class lcstsdataset(Dataset):
    def __init__(self,text_path):
        lst=[]
        for fp in open(text_path,'r',encoding = 'utf-8'):
            data = json.loads(fp)
            lst.append(data)
        self.data_list=lst

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]['input'][29:],self.data_list[idx]['output']#不考虑前29个提示