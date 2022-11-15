import pandas as pd
source = pd.read_csv("/home/linzhisheng/esg/source.csv", nrows = 1000000)
source = source[source['Abstract'].notna()]

source = source[['Abstract']]
source['Abstract'] = source['Abstract'].replace('\t',' ', regex=True).replace('\n',' ', regex=True)
print(source)
source.to_csv("source_100sw", index=False)
