# 文档检索管理系统

## 系统概述
本系统是一个基于云端的文档资料统一管理平台，实现文档的集中管
理和知识元的统一归档。在该系统中，用户可以方便地上传、下载和共享历史案
例资料，包括各种技术文档、相关规范等。系统支持帮助用户快速查找和检索所
需的信息。此外，系统还支持对于知识元的管理，知识元和技术方案唯一绑定。
该系统还拥有技术方案自动生成的功能。通过算法分析，系统能够在新建工程项
目时，通过和历史案例资料的相似度对比，自动生成技术方案。
  
## 生成摘要
```
python predict.py --model_path /path/to/model --file_path /path/to/file/ --file__type 1 --svr_dir ./svr
```  

## 模型  
[**CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation**](https://arxiv.org/pdf/2109.05729.pdf)  
CPT模型的源码地址：https://github.com/fastnlp/CPT
  
  ## 数据集
  [**Lcsts**](https://huggingface.co/datasets/suolyer/lcsts)

## 多文本摘要算法参考论文
[**Adapting the Neural Encoder-Decoder Framework from Single to Multi-Document Summarization**]
(https://arxiv.org/pdf/1808.06218)  

### Requirements:
- pytorch==1.8.1
- transformers==4.4.1
- packaging==21.3
- rouge-chinese
- numpy==1.20.1
- scikit-learn==0.24.1