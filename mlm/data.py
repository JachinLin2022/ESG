import pandas as pd
import re

def remove_notna():
    source = pd.read_csv("/home/linzhisheng/esg/source.csv")
    source = source[source['Abstract'].notna()]

    source = source[['Abstract']]
    source['Abstract'] = source['Abstract'].replace('\t',' ', regex=True).replace('\n',' ', regex=True)
    print(source)
    source.to_csv("source_all", index=False)

def remove_upprintable_chars(s):
    """移除所有不可见字符"""
    return ''.join(x for x in s if x.isprintable())
def remove_unseen_and_spaec():
    source = pd.read_csv("/home/linzhisheng/esg/mlm/source_all")
    source['Abstract'] = source['Abstract'].map(lambda x: re.sub(r" +", ' ', remove_upprintable_chars(x)))
    source['Label'] = ''
    source.to_csv("source_all_format", index=False)


remove_unseen_and_spaec()