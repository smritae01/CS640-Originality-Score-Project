"""
This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""
import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

from model import GPT2PPLV2 as GPT2PPL
import pandas as pd
import csv

print(torch.cuda.is_available())
# def free_gpu_cache():
#     print("Initial GPU Usage")
#     gpu_usage()                             

#     torch.cuda.empty_cache()

#     cuda.select_device(0)
#     cuda.close()
#     cuda.select_device(0)

#     print("GPU Usage after emptying the cache")
#     gpu_usage()

# free_gpu_cache()                  

# initialize the model
model = GPT2PPL()

df = pd.read_csv("dataset.csv",encoding = "ISO-8859-1")
# scores = []
# c=0
# for i in df["Text"]:
#     scores.append(model(i, 100, "v1.1"))
#     c+=1
#     print(c)
# print(scores)

with open('zscores4-hum.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    for i in df["Text"]:
        score=model(i, 100, "v1.1")
        writer.writerow([score])