# 项目名称：ESG信息抽取

ESG信息抽取项目旨在通过自然语言处理和机器学习技术，自动从企业社会责任报告、媒体报道等文档中提取ESG（环境、社会和治理）相关的指标信息，并进行分析和汇总，为用户提供ESG评级、趋势分析、竞争对手比较等功能。

## 目录结构

- `main.py`：读取全文报告，滑动窗口并进行评价
- `train.py`：读取训练集和测试集，训练短句模型并保存
- `eval.py`：在短句测试集上评价模型
- `data.py, process_data.ipynb`：数据预处理工具代码
- `chatgpt.py`：使用chatgpt api来评价chatgpt准确率
- `analyse.ipynb, analyse2.ipynb`：数据分析、准确率统计代码
- `report_scrapy`：全文报告爬虫


## 数据源

本次数据来源于Refinitiv，该公司从企业年报、NGO网站、CSR报告、公司网站、新闻媒体等渠道收集了Environment, Social and Governance相关的数据，并对这些数据进行了数据分析。

## 环境要求

该项目需要使用Python语言和一些常见的Python科学计算库。以下是本项目的环境要求：

* Python
* Pytorch 
* NumPy
* Pandas
* PyMuPDF pdf转文本
* pdfplumber pdf表格提取

## 数据预处理

数据只包含短句不包含全文报告，经过对数据集进行预处理以及全文报告爬取后，得到的数据规模为：
* 训练集：27000条短句
* 测试集：4509条短句  4509篇全文报告
  
模型在训练集短句中进行训练，训练完毕后分别在测试集短句以及全文报告中进行评价。由于ESG报告全文过长无法一次性输入到模型当中，使用滑动窗口的方式将全文报告划分为一个个子句再输入到模型当中，最终的全文报告生成的数据规模为：
* 窗口大小128，步长32，3426388个窗口
* 窗口大小384，步长128，1287602个窗口

## 模型

本次任务属于Extractive QA，输入为子句+问题，问题的答案在子句当中。选择微调的模型为：bert-large-uncased-whole-word-masking-finetuned-squad，该模型已经在squad预训练好，具有一定的信息提取能力。


## 结果分析

## 对比chatgpt

| **测试集规模：3932**   | **match**   | **time(s)** |
|------------------|-------------|-------------|
| chatgpt prompt 1 | 0.475076297 | 8,436       |   |
| chatgpt prompt 2 | 0.460278    | 6,538       |
| bert-large-esg-fine-tune            | 0.7285429141716567  | 56          |
| deberta-large-esg-fine-tune            | 0.739853626081171  |         |


## deberta-large-v3
  
| **deberta v3**                           | **prompt1 topn=10** | **prompt2 topn=10** | **表格抽取topn=10** | **prompt2 topn=10+表格抽取** | **结合两种prompt topn=10+表格抽取** |
|------------------------------------------|---------------------|---------------------|-----------------|--------------------------|-----------------------------|
| **ALL,find**                             | 0.781134686         | 0.794280443         | 0.294972325     | 0.825876384              | 0.865313653                 |
| **ElectricityPurchased**                 | 0.886509636         | 0.892933619         | 0.308351178     | 0.907922912              | 0.931477516                 |
| **EnergyUseTotal**                       | 0.798219585         | 0.774480712         | 0.210682493     | 0.810089021              | 0.839762611                 |
| **WasteRecycledTotal**                   | 0.808333333         | 0.804166667         | 0.241666667     | 0.841666667              | 0.904166667                 |
| **HazardousWaste**                       | 0.735               | 0.745               | 0.29            | 0.775                    | 0.815                       |
| **RenewableEnergyProduced**              | 0.884816754         | 0.853403141         | 0.267015707     | 0.890052356              | 0.921465969                 |
| **DonationsTotal**                       | 0.597826087         | 0.706521739         | 0.168478261     | 0.733695652              | 0.777173913                 |
| **EnergyPurchasedDirect**                | 0.770491803         | 0.737704918         | 0.289617486     | 0.770491803              | 0.841530055                 |
| **CO2EquivalentsEmissionDirectScope1**   | 0.774011299         | 0.790960452         | 0.344632768     | 0.813559322              | 0.847457627                 |
| **CO2EquivalentsEmissionIndirectScope3** | 0.564102564         | 0.679487179         | 0.326923077     | 0.75                     | 0.769230769                 |
| **ElectricityProduced**                  | 0.821917808         | 0.787671233         | 0.253424658     | 0.808219178              | 0.890410959                 |
| **NonHazardousWaste**                    | 0.622047244         | 0.724409449         | 0.228346457     | 0.732283465              | 0.787401575                 |
| **WasteTotal**                           | 0.745901639         | 0.721311475         | 0.18852459      | 0.754098361              | 0.803278689                 |
| **NOxEmissions**                         | 0.919642857         | 0.928571429         | 0.348214286     | 0.955357143              | 0.973214286                 |
| **TotalInjuryRateTotal**                 | 0.892857143         | 0.910714286         | 0.383928571     | 0.955357143              | 0.955357143                 |
| **TotalInjuryRateEmployees**             | 0.902912621         | 0.912621359         | 0.359223301     | 0.932038835              | 0.932038835                 |



## bert-large
  
|**tpye**                             | **prompt1 topn=10** | **prompt2 topn=10** | **表格抽取topn=10** | **prompt2 topn=10+表格抽取** | **结合两种prompt topn=10+表格抽取** |
|--------------------------------------|---------------------|---------------------|-----------------|--------------------------|-----------------------------|
| ALL,find                             | 0.766143911         | 0.76199262          | 0.283440959     | 0.803736162              | 0.857933579                 |
| ElectricityPurchased                 | 0.888650964         | 0.901498929         | 0.301927195     | 0.907922912              | 0.935760171                 |
| EnergyUseTotal                       | 0.786350148         | 0.765578635         | 0.210682493     | 0.807121662              | 0.845697329                 |
| WasteRecycledTotal                   | 0.7875              | 0.8125              | 0.225           | 0.833333333              | 0.883333333                 |
| HazardousWaste                       | 0.695               | 0.765               | 0.28            | 0.735                    | 0.845                       |
| RenewableEnergyProduced              | 0.869109948         | 0.821989529         | 0.261780105     | 0.90052356               | 0.90052356                  |
| DonationsTotal                       | 0.739130435         | 0.679347826         | 0.157608696     | 0.766304348              | 0.804347826                 |
| EnergyPurchasedDirect                | 0.786885246         | 0.803278689         | 0.284153005     | 0.836065574              | 0.885245902                 |
| CO2EquivalentsEmissionDirectScope1   | 0.700564972         | 0.762711864         | 0.282485876     | 0.740112994              | 0.81920904                  |
| CO2EquivalentsEmissionIndirectScope3 | 0.557692308         | 0.673076923         | 0.288461538     | 0.634615385              | 0.762820513                 |
| ElectricityProduced                  | 0.842465753         | 0.794520548         | 0.267123288     | 0.863013699              | 0.904109589                 |
| NonHazardousWaste                    | 0.582677165         | 0.677165354         | 0.220472441     | 0.637795276              | 0.771653543                 |
| WasteTotal                           | 0.737704918         | 0.713114754         | 0.213114754     | 0.770491803              | 0.81147541                  |
| NOxEmissions                         | 0.901785714         | 0.839285714         | 0.348214286     | 0.928571429              | 0.946428571                 |
| TotalInjuryRateTotal                 | 0.857142857         | 0.901785714         | 0.375           | 0.919642857              | 0.964285714                 |
| TotalInjuryRateEmployees             | 0.902912621         | 0.825242718         | 0.310679612     | 0.912621359              | 0.922330097                 |

* exact-match 模型输出答案和正确答案一致

| **tpye**                             | **prompt1 topn=10** | **prompt2 topn=10** | **表格抽取topn=10** | **prompt2 topn=10+表格抽取** | **结合两种prompt topn=10+表格抽取** |
|--------------------------------------|---------------------|---------------------|-----------------|--------------------------|-----------------------------|
| ALL                                  | 0.687730627         | 0.679658672         | 0.227167897     | 0.718634686              | 0.776983395                 |
| ElectricityPurchased                 | 0.837259101         | 0.845824411         | 0.274089936     | 0.862955032              | 0.882226981                 |
| EnergyUseTotal                       | 0.735905045         | 0.724035608         | 0.175074184     | 0.756676558              | 0.804154303                 |
| WasteRecycledTotal                   | 0.695833333         | 0.7375              | 0.1625          | 0.775                    | 0.808333333                 |
| HazardousWaste                       | 0.595               | 0.665               | 0.245           | 0.705                    | 0.76                        |
| RenewableEnergyProduced              | 0.848167539         | 0.790575916         | 0.235602094     | 0.832460733              | 0.884816754                 |
| DonationsTotal                       | 0.652173913         | 0.597826087         | 0.108695652     | 0.625                    | 0.706521739                 |
| EnergyPurchasedDirect                | 0.732240437         | 0.726775956         | 0.25136612      | 0.781420765              | 0.825136612                 |
| CO2EquivalentsEmissionDirectScope1   | 0.621468927         | 0.689265537         | 0.248587571     | 0.706214689              | 0.745762712                 |
| CO2EquivalentsEmissionIndirectScope3 | 0.487179487         | 0.596153846         | 0.25            | 0.647435897              | 0.698717949                 |
| ElectricityProduced                  | 0.787671233         | 0.719178082         | 0.219178082     | 0.732876712              | 0.828767123                 |
| NonHazardousWaste                    | 0.503937008         | 0.566929134         | 0.188976378     | 0.590551181              | 0.653543307                 |
| WasteTotal                           | 0.68852459          | 0.62295082          | 0.180327869     | 0.655737705              | 0.737704918                 |
| NOxEmissions                         | 0.821428571         | 0.714285714         | 0.276785714     | 0.803571429              | 0.883928571                 |
| TotalInjuryRateTotal                 | 0.776785714         | 0.821428571         | 0.330357143     | 0.875                    | 0.892857143                 |
| TotalInjuryRateEmployees             | 0.815533981         | 0.708737864         | 0.262135922     | 0.72815534               | 0.825242718                 |

## Reference



IR相关论文：
1. Rapid Adaptation of BERT for Information Extraction on Domain-Specific Business Documents (https://arxiv.org/pdf/2002.01861.pdf) 这篇论文讨论了如何使用BERT从商业文件（如合同、报表和备案）中提取重要内容元素。
2. Jointly Learning Span Extraction and Sequence Labeling for Information Extraction from Business Documents (https://ieeexplore.ieee.org/abstract/document/9892779) 本文介绍了一种新的针对商业文件的信息提取模型。不同于以往研究仅基于跨度提取或序列标注，该模型充分利用了两种方法的优势。该组合使得该模型能够处理信息稀疏的长文档（提取的信息量很小）。
3. Gain more with less: Extracting information from business documents with small dat (https://www.sciencedirect.com/science/article/pii/S0957417422022928)
4. Transformers-based information extraction with limited data for domain-specific business documents (https://www.sciencedirect.com/science/article/pii/S0952197620303481)
   







Extractive QA相关：

1. Choose Your QA Model Wisely: A Systematic Study of Generative and Extractive Readers for Question Answering (https://arxiv.org/abs/2203.07522) 研究了抽取式和生成式在问答任务中的比较。研究人员发现，生成式在长文本问答中表现更好，而抽取式在短文本中表现更好，并且在领域外泛化方面也表现更好。

2. "Reading Wikipedia to Answer Open-Domain Questions" (https://arxiv.org/abs/1704.00051) - 这篇论文介绍了一种基于阅读维基百科的方法进行开放域的Extractive QA。


3. "DrQA" (https://github.com/facebookresearch/DrQA) - 这个github库是DrQA的官方实现，包含用于Extractive QA的Python代码。

2. "SQuAD" (https://github.com/rajpurkar/SQuAD-explorer) - 这个github库包含SQuAD数据集的浏览器，SQuAD是一种用于Extractive QA模型评估的数据集。

表格处理相关：
1. pdfplumber: Plumb a PDF for detailed information about each text character, rectangle, and line. Plus: Table extraction and visual debugging. (https://github.com/JachinLin2022/ESG/tree/master/QA)