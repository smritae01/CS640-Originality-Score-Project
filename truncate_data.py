import pandas as pd
from transformers import AutoTokenizer
from transformers import GPT2Tokenizer
import re

df = pd.read_csv('./data/GPT-wiki-intro.csv')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

max_tokens = 245

# did it onky for the first 501 rows
for i in range(500):

    intro_text = df.loc[i, 'wiki_intro']

    tokens = tokenizer.encode(intro_text, add_special_tokens=False)
    
    if len(tokens) > max_tokens:
        text = tokenizer.decode(tokens[:max_tokens], clean_up_tokenization_spaces=True)
        
        text = re.sub(r'\s\S*$', '', text)
        
        df.loc[i, 'wiki_intro'] = text

df = df.iloc[:500]

df.to_csv('./data/updated_wiki.csv', index=False)
