# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import pandas as pd
import os
import time
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
import openai
key = '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'
openai.api_key = key
data = pd.read_csv('test.csv')
data['chat'] = ''
print(data)
T1 = time.time()
# count = 738
for index,row in data.iterrows():
    # if index < 1601:
    #     continue
    # print(index)
    t1 = time.time()
    text = row['text']
    path = row['path']
    value = row['value']
    
    rsp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "user", "content": text + '\n' + 'Extract origin number about {}.'.format(path)},
        ]
    )
    result = rsp['choices'][0]['message']['content']
    data.loc[index, 'chat'] = result
    # row['chat'] = result
    print((index, value, result))
    if result.find(value) >= 0:
        count = count + 1
    # print(path)
    if index % 50 == 0:
        print(count)
        T2 = time.time()
        print('程序运行时间:%s毫秒' % ((T2 - T1)))
        data.to_csv('res_prompt2_2.csv',index=False)

    
    # time.sleep(15)
    t2 = time.time()
    print(t2-t1)
    
T2 = time.time()
print(data['chat'][0])
print('程序运行时间:%s毫秒' % ((T2 - T1)))
print(count)
