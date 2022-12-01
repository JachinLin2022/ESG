# Update
## Reference
1. NLP预训练中的mask方式总结 https://zhuanlan.zhihu.com/p/434672623
2. MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction https://aclanthology.org/2022.findings-acl.34.pdf
3. zero-shot learning https://joeddav.github.io/blog/2020/05/29/ZSL.html
4. FinBERT: Financial Sentiment Analysis with Pre-trained Language Models https://arxiv.org/pdf/1908.10063.pdf
5. SCI BERT: A Pretrained Language Model for Scientific Text  https://arxiv.org/pdf/1903.10676.pdf
6. Domain-Specific BERT Models https://mccormickml.com/2020/06/22/domain-specific-bert-tutorial/#3-comparing-scibert-and-bert

## 2022/12/1
1. 粗粒度分类：
```
0                    CommunityDataPoints  113515329
10       ProductResponsibilityDataPoints  112224442
12                   WorkforceDataPoints  111043370
22                 HumanRightsDataPoints  115004608
1089754               EmissionDataPoints  113162528
1089758             InnovationDataPoints  130945423
1089760            ResourceUseDataPoints  115004607
1709643             ManagementDataPoints  114003098
1709644           ShareholdersDataPoints  111498446
1709646            CsrStrategyDataPoints  113164169
```
对有标签的数据提取词表，做成训练集，100w
1. 修改Mask策略重新训练200w，对于选定的词性单词，80%情况下进行Whole Word Mask，20%情况下保持原词汇。
  + 基于模板的关键词不一定在原句中出现
  + 使用KerBert+预训练embedding层抽取的关键字是原句出现的
2. 基于ESG语料训练roberta tokenizer
  + 使用原始roberta tokenizer，选取前top 50000个词，计算获得关键词占原句单词个数的比例，100条平均比例为0.601，原因是roberta-large原始vocab当中不包含原句中的部分单词
```
top_tokens = torch.topk(mask_token_logits, 50000, dim=1).indices[0].tolist()

['interpublics', 'directors', 'are', 'elected', 'each', 'year', 'by', 'interpublics', 'stockholders', 'at', 'the', 'annual', 'meeting', 'of', 'stockholders', 'interpublics', 'corporate', 'governance', 'committee', 'recommends', 'nominees', 'to', 'the', 'board', 'of', 'directors', 'and', 'the', 'board', 'proposes', 'a', 'slate', 'of', 'nominees', 'to', 'the', 'stockholders', 'for', 'election']
-----------------------------------
 The Keyword is <mask>.:intersect:{'year', 'board', 'for', 'elected', 'a', 'are', 'each', 'to', 'annual', 'recommends', 'at', 'directors', 'nominees', 'by', 'and', 'governance', 'the', 'corporate', 'slate', 'of', 'committee', 'election', 'meeting'}, radio is 0.5897435897435898

 The keyword is <mask>.:intersect:{'year', 'board', 'for', 'elected', 'a', 'are', 'each', 'to', 'annual', 'recommends', 'at', 'directors', 'nominees', 'by', 'and', 'governance', 'the', 'corporate', 'slate', 'of', 'committee', 'election', 'proposes', 'meeting'}, radio is 0.6153846153846154

 In summary, the related word is <mask>.:intersect:{'year', 'board', 'for', 'elected', 'a', 'are', 'each', 'to', 'annual', 'recommends', 'at', 'directors', 'nominees', 'by', 'and', 'governance', 'the', 'corporate', 'slate', 'of', 'committee', 'election', 'proposes', 'meeting'}, radio is 0.6153846153846154

 In summary, the keyword is <mask>.:intersect:{'year', 'board', 'for', 'elected', 'a', 'are', 'each', 'to', 'annual', 'recommends', 'at', 'directors', 'nominees', 'by', 'and', 'governance', 'the', 'corporate', 'slate', 'of', 'committee', 'election', 'proposes', 'meeting'}, radio is 0.6153846153846154

 <mask> is the keyword.:intersect:{'year', 'board', 'for', 'a', 'are', 'each', 'to', 'annual', 'at', 'directors', 'nominees', 'by', 'and', 'the', 'governance', 'corporate', 'slate', 'of', 'committee', 'election', 'proposes', 'meeting'}, radio is 0.5641025641025641

 In summary, <mask> is the related word.:intersect:{'year', 'board', 'for', 'elected', 'a', 'are', 'each', 'to', 'annual', 'recommends', 'at', 'directors', 'nominees', 'by', 'and', 'governance', 'the', 'corporate', 'slate', 'of', 'committee', 'election', 'proposes', 'meeting'}, radio is 0.6153846153846154
```
  + 在ESG语料训练tokenizer，vocab size=50265，basevocab与esg vocab的intersection = 0.547，关键词占原句单词个数的比例为0.622
```
['Inter', 'public', 's', 'ĠDirectors', 'Ġare', 'Ġelected', 'Ġeach', 'Ġyear', 'Ġby', 'ĠInter', 'public', 's', 'Ġstock', 'holders', 'Ġat', 'Ġthe', 'Ġannual', 'Ġmeeting', 'Ġof', 'Ġstock', 'holders', '.', 'ĠInter', 'public', 's', 'ĠCorporate', 'ĠGovern', 'ance', 'ĠCommittee', 'Ġrecommends', 'Ġnominees', 'Ġto', 'Ġthe', 'ĠBoard', 'Ġof', 'ĠDirectors', ',', 'Ġand', 'Ġthe', 'ĠBoard', 'Ġproposes', 'Ġa', 'Ġslate', 'Ġof', 'Ġnominees', 'Ġto', 'Ġthe', 'Ġstock', 'holders', 'Ġfor', 'Ġelection', '.']
['Inter', 'public', 's', 'ĠDirectors', 'Ġare', 'Ġelected', 'Ġeach', 'Ġyear', 'Ġby', 'ĠInterpublic', 's', 'Ġstockholders', 'Ġat', 'Ġthe', 'Ġannual', 'Ġmeeting', 'Ġof', 'Ġstockholders', '.', 'ĠInterpublic', 's', 'ĠCorporate', 'ĠGovernance', 'ĠCommittee', 'Ġrecommends', 'Ġnominees', 'Ġto', 'Ġthe', 'ĠBoard', 'Ġof', 'ĠDirectors', ',', 'Ġand', 'Ġthe', 'ĠBoard', 'Ġproposes', 'Ġa', 'Ġslate', 'Ġof', 'Ġnominees', 'Ġto', 'Ġthe', 'Ġstockholders', 'Ġfor', 'Ġelection', '.']
```
  + 原始roberta-large与esg-roberta-large对比
```
To contribute to climate change mitigation, we actively explore opportunities to support local renewable energy generation. Solar panels are installed at Hang Seng 113 to generate renewable energy.

#原始roberta-large 生成的词表比较泛化
['green', 'sustainability', 'sustainable', 'energy', 'mitigation', 'renewable', 'solar', 'sustainable', 'green', 'environment', 'clean', 'energy', 'renewables', 'environmental', 'carbon', 'electricity', 'welcome', 'climate', 'zero', 'environment', 'green', 'adaptation', 'wind', 'proactive', 'innovation', 'resilience', 'power', 'positive', 'conservation', 'transparency', 'responsible', 'commitment', 'carbon', 'this', 'solar', 'here', 'greenhouse', 'that', 'inclusive', 'ambition', 'climate', 'resilient', 'awareness', 'support', 'emissions', 'pollution', 'coal', 'global', 'mitigating', 'local', 'opportunity', 'relevant', 'action', 'simple', 'effective', 'important', 'innovative', 'technology', 'efficiency', 'investment', 'key', 'water', 'biomass', 'impact', 'enable', 'future', 'fuel', 'done', 'balance', 'enabling', 'transparent', 'will', 'development', 'infrastructure', 'community', 'help', 'the', 'smart', 'appropriate', 'ecosystem', 'collaborative', 'friendly', 'emission', 'waste', 'power', 'change', 'possible', 'sunshine', 'significant', 'responsibility', 'ambitious', 'solution', 'efficient', 'solutions', 'policy', 'growth', 'recycling', 'act', 'active', 'everywhere']


#100w ramdom mask 生成的词表能够精确到energy领域
['renewable', 'energy', 'electricity', 'wind', 'water', 'clean', 'green', 'biomass', 'solar', 'coal', 'carbon', 'nonrenewable', 'recycled', 'steam', 'power', 'heat', 'lower', 'low', 'conventional', 'biogas', 'hydroelectric', 'photovoltaic', 'heating', 'thermal', 'cleaner', 'wood', 'fuel', 'geothermal', 'recovered', 'zero', 'lighting', 'diagnostics', 'electric', 'paper', 'packaging', 'natural', 'improved', 'cement', 'reduced', 'incineration', 'purchased', 'rainwater', 'energysaving', 'less', 'new', 'material', 'used', 'gas', 'recyclable', 'oil', 'dust', 'emissions', 'more', 'reliable', 'diesel', 'sustainable', 'small', 'tree', 'grid', 'fossil', 'affordable', 'wind', 'hydropower', 'wastewater', 'nuclear', 'installed', 'large', 'generated', 'hydrocarbons', '60', 'household', 'total', 'reusable', 'waste', 'renewable', 'global', 'ghg', 'production', 'travel', 'steel', 'renewables', 'discharged', 'generation', 'cooling', 'hydro', 'economical', 'freshwater', 'petrol', 'equivalent', 'transport', 'current', 'hot', 'exhaust', 'concrete', 'co', 'greenhouse', 'cutting', 'efficiently', 'alternative', 'air']
```
4. 针对词表的特征向量
5. 使用hdbscan对词表进行聚类
  + 使用预训练好的模型提取sentence embedding以及word embedding
  + 使用umap进行降维 n_neighbors=15, n_components=5，15
  + 使用hdbscan进行聚类min_cluster_size=5，10，15
  + 聚类二维可视化
  + tsne   距离cos  皮尔森相关系数
## 2022/11/23
1. 使用spacy库Mask 一半数据，需要跑半天时间
2. 由于机器性能的问题，先在一半的数据，200w条mask文本上进行了训练，需要跑三天时间
```
 The keyword is <mask>.:['green', 'sustainable', 'renewable', 'solar', 'sustainability', 'zero', 'clean', 'GREEN', 'below', 'here', 'energy', 'ongoing', 'Sustainable', 'welcome', 'free', 'local', 'Green', 'Solar', 'low', 'transparent', 'positive', 'available', 'affordable', 'above', 'sunlight', 'simple', 'used', 'wind', 'sun', '2030', 'efficient', 'distributed', 'PV', 'scalable', 'required', 'recycling', 'blue', 'conservation', 'bright', 'recycled', 'inclusive', 'unique', 'renewables', 'included', 'LED', 'continuous', 'Zero', 'effective', 'growing', 'YES']

 In summary, the related word is <mask>.:['sustainability', 'sustainable', 'green', 'solar', 'renewable', 'energy', 'Sustainable', 'zero', 'here', 'GREEN', 'mitigation', 'below', 'electricity', 'welcome', 'clean', 'wind', 'greenhouse', 'this', 'biomass', 'Solar', 'renewables', 'that', 'carbon', 'Green', 'Energy', 'environmental', 'positive', 'available', 'used', 'above', 'done:', 'unsustainable', 'simple', 'waste', 'important', 'power', 'effective', 'conservation', 'climate', 'water', 'sun', 'possible', 'achieved', 'abundant', 'efficiency', 'required', 'ongoing', 'significant', 'key']
 
--------------------------------------

In summary, the related word is <mask>.:['green', 'energy', 'sustainable', 'sustainability', 'renewable', 'environment', 'environmental', 'GREEN', 'mitigation', 'Energy', 'Green', 'carbon', 'climate', 'relevant', 'positive', 'solar', 'electricity', 'support', 'zero', 'proactive', 'renewables', 'greenhouse', 'Environment', 'reduce', 'Sustainable', '', 'neutral', 'responsible', 'local', 'key', 'adaptation', 'Carbon', 'possible', 'enable', 'emission', 'appropriate', 'target', 'emissions', 'clean', 'innovative', 'important', 'global', 'ECO', 'essential', 'YES', 'active', 'friendly....', 'opportunities', 'mitigating']

 In summary, the keyword is <mask>.:['green', 'sustainability', 'zero', 'sustainable', 'mitigation', 'GREEN', 'energy', 'carbon', 'proactive', 'positive', 'renewable', 'yes', 'Yes', 'YES', 'Green', 'change', 'Zero', 'reduce', 'renewables', 'neutral', 'action', 'environmental', 'low', 'Carbon', 'mitigating', 'key', 'adaptation', 'target', 'reduction', 'Reduce', 'local', 'climate', 'future', 'relevant', 'conservation', 'good', 'avoided', 'active', 'clean', 'responsible', 'environment', 'progress', 'efficiency', 'balance', 'focus', 'there', 'no', 'Sustainable', 'innovation', 'C']
 
 -----------------------------------------------------

The keyword is <mask>.:['green', 'sustainable', 'renewable', 'clean', 'solar', 'local', 'Green', 'GREEN', 'positive', 'sustainability', 'friendly', 'environmental', 'Sustainable', 'low', 'neutral', 'Solar', 'Local', 'global', 'simple', 'responsible', 'efficient', 'avoided', 'zero', 'Clean', 'conservation', 'natural', 'recycling', 'good', 'cleaner', 'innovative', 'efficiency', 'Go', 'smart', 'recycled', 'ongoing', 'circular', 'saving', 'internal', 'environment', 'energy', 'new', 'community', 'continued', 'focus', 'associate', 'future', 'affordable', 'Renew', 'key', 'distributed']

In summary, the related word is <mask>.:['avoided', 'green', 'achieved', 'used', 'encouraged', 'done', 'recycled', 'continued', 'included', 'provided', 'implemented', 'embraced', 'renewable', 'applied', 'introduced', 'sustainable', 'supported', 'adopted', 'recommended', 'linked', 'growing', 'connected', 'GREEN', 'promoted', 'completed', 'welcomed', 'positive', 'utilized', 'advocated', 'Green', 'working', 'increasing', 'simplified', 'rising', 'reduced', 'added', 'related', 'made', 'addressed', 'reused', 'considered', 'saved', 'established', 'key', 'highlighted', 'chosen', 'checked', 'mentioned', 'reviewed', 'delivered']
```
1. 下游ESG分类任务，将预测词表组合成字符串句子，输入到zero-shot分类微调后的模型当中，使用类别概率作为评估结果
  1. 使用facebook/bart-large-mnli作为baseline，计算原句子的概率结果
    1. NLI premise and to construct a hypothesis from each candidate label. For example, if we want to evaluate whether a sequence belongs to the class "politics", we could construct a hypothesis of This text is about politics.. The probabilities for entailment and contradiction are then converted to label probabilities.
    2. https://joeddav.github.io/blog/2020/05/29/ZSL.html
  2. 使用ESG微调后，不同模板生成的单词组成句子，计算该句子的概率结果
    1. 对比baseline和esg微调模型
    2. 对比不同的模板
    3. 对比不同mask
```
baseline: {'sequence': 'Interpublics Directors are elected each year by Interpublics stockholders at the annual meeting of stockholders. Interpublics Corporate Governance Committee recommends nominees to the Board of Directors, and the Board proposes a slate of nominees to the stockholders for election.', 'labels': ['company', 'social', 'environment'], 'scores': [0.9741765856742859, 0.17715317010879517, 0.15736056864261627]}

template: In summary, the related word is <mask>. res:{'sequence': 'Director Chairman director independent Board governance chairman Directors directors President elected independence Chair CEO president Independent above appointed required important Secretary appointment mandatory below fundamental elect necessary voting election imperative Corporate nominee effective Corporation Govern nomination essential board nominees incumbent management nominated Company Management control corporate Leadership officers Officer Governor ', 'labels': ['company', 'social', 'environment'], 'scores': [0.9887624382972717, 0.5372319221496582, 0.5047216415405273]}

template: In summary, the keyword is <mask>. res:{'sequence': 'independent Chairman Independent chairman Directors Director director directors President elected CEO Board president management appointed Management Chair continuous annual below nonpartisan above nominated permanent incumbent Corporate shareholders diverse independence current governance board here voting effective certain recommended appointment unknown new nominees ongoing required compensation approved responsible internal human corporate necessary ', 'labels': ['company', 'social', 'environment'], 'scores': [0.9798027873039246, 0.21766702830791473, 0.06311383843421936]}

template: <mask> is the keyword. res:{'sequence': 'Below This Here Voting That Chairman Following Disclosure It As Today Information Such There Compensation Above Management Diversity Transparency Change here Election Deadline Registration Who Summary belowThis What Director He ParticipationBelow President HERE Date Retirement Staff Committee ThusHere One Compliance Strategy Name CEO So Finance Each Membership ', 'labels': ['company', 'social', 'environment'], 'scores': [0.9686070680618286, 0.6541833281517029, 0.11164838820695877]}

template: In summary, the related word is <mask>. res:{'sequence': 'Board Director elected nomination director election Directors independent nominees nominee Committee committee nominated Company governance Chairman independence diversity directors candidates board Diversity term inclusive process persons incumbent individual candidate appointment qualified Independent all applicable Boards Election nominating Vote vote nominations appointed nominate elect voting required continuous person name Yes committees ', 'labels': ['company', 'social', 'environment'], 'scores': [0.9612820148468018, 0.4090128242969513, 0.3390839993953705]}

template: In summary, the keyword is <mask>. res:{'sequence': 'effective outstanding diversity strong performance excellent inclusive Yes accountability independent fair transparent Board transparency integrity good governance independence quality balance Diversity responsive comprehensive balanced nominees efficiency Directors elected compliance process engaged business accountable efficient yes all consistent Company clear oversight Independent engagement leadership open Effective teamwork consistency qualified people Strong ', 'labels': ['company', 'social', 'environment'], 'scores': [0.9898789525032043, 0.6757798790931702, 0.4573588967323303]}

template: <mask> is the keyword. res:{'sequence': 'Transparency Diversity Performance Process Quality Accountability Integrity Voting Competition Independence Innovation Effective Choice Fair Efficiency Evaluation Compliance Board Excellence Election Trust Responsibility Participation Balanced Experience Independent Committee Director Selection Conduct Execution Accuracy Leadership Oversight Success Management Vote This Team Balance Improvement Period Progressive Action All Ethics Opportunity Unity Candidate Best ', 'labels': ['social', 'company', 'environment'], 'scores': [0.7891299724578857, 0.5237616300582886, 0.38252368569374084]}

template: In summary, the related word is <mask>. res:{'sequence': 'elected chosen preferred cumulative staggered appointed approved Board voting committee recommended process done established selected nominated qualified election said defined used governed governance annual adopted linked proposed board decided voted determined Committee required named reelection connected confirmed integrated simple Committees considered delegated agreed attached key introduced separated applied formed continuous ', 'labels': ['company', 'social', 'environment'], 'scores': [0.6822032332420349, 0.5789298415184021, 0.23069503903388977]}

template: In summary, the keyword is <mask>. res:{'sequence': 'process accountability transparency governance quality balance consistency alignment integrity transparent good people clear balanced fair diversity choice consistent simple oversight trust engagement retention Accountability democracy fairness effective Board composition Process committee teamwork done processes voting check collaboration system best Balanced comprehensive right principle directors performance efficiency coordination routine change democratic ', 'labels': ['company', 'social', 'environment'], 'scores': [0.7720822691917419, 0.7342028021812439, 0.090058833360672]}

template: <mask> is the keyword. res:{'sequence': 'Process Accountability Quality Transparency Choice Balance Procedure Diversity Participation Committee Planning Efficiency Responsibility Principle Innovation Excellence Voting Performance Progress Committees Independence Competition Focus Priority Structure Execution This Action Leadership Implementation Partnership Board Selection Management Team Experience That Order Compliance Precision Trust It Organization Integrity System Detail Input Oversight Everyone process ', 'labels': ['company', 'social', 'environment'], 'scores': [0.7668386697769165, 0.29952892661094666, 0.1309095174074173]}
baseline: {'sequence': 'To contribute to climate change mitigation, we actively explore opportunities to support local renewable energy generation. Solar panels are installed at Hang Seng 113 to generate renewable energy.', 'labels': ['environment', 'company', 'social'], 'scores': [0.9960302114486694, 0.9253985285758972, 0.23369087278842926]}

#原始模型
template: In summary, the related word is <mask>. res:{'sequence': 'sustainability sustainable green solar renewable energy Sustainable zero here GREEN mitigation below electricity welcome clean wind greenhouse this biomass Solar renewables that carbon Green Energy environmental positive available used above done: unsustainable simple waste important power effective conservation climate water sun possible achieved abundant efficiency required ongoing significant key ', 'labels': ['environment', 'social', 'company'], 'scores': [0.9766672849655151, 0.3054905831813812, 0.1270236223936081]}

template: In summary, the keyword is <mask>. res:{'sequence': 'green sustainable sustainability renewable zero clean energy positive solar GREEN here simple renewables there low working Green clear everywhere Sustainable mitigation ongoing transparent done local below welcome conservation good available change YES no resilience effective yes environmental carbon resilient water abundant achievable this affordable bright consistent efficient progress transparency balance ', 'labels': ['environment', 'social', 'company'], 'scores': [0.9917229413986206, 0.22152167558670044, 0.1315782219171524]}

template: <mask> is the keyword. res:{'sequence': 'Efficiency Green Solar Sustainable Clean Energy This Environment Conservation Renew Innovation Sunshine Climate Environmental Future Zero That Carbon GREEN Wind Local Success Sun Light Positive Change Tomorrow Development Alternative Responsibility Focus Community Shade Storage Solutions sustainability It Technology Transparency Simple Diversity Action Affordable Savings Growth Awareness Protection Education Source Growing ', 'labels': ['environment', 'social', 'company'], 'scores': [0.9901493787765503, 0.7556861639022827, 0.037533942610025406]}

#random mask
template: In summary, the related word is <mask>. res:{'sequence': 'green energy sustainable sustainability renewable environment environmental GREEN mitigation Energy Green carbon climate relevant positive solar electricity support zero proactive renewables greenhouse Environment reduce Sustainable  neutral responsible local key adaptation Carbon possible enable emission appropriate target emissions clean innovative important global ECO essential YES active friendly.... opportunities mitigating ', 'labels': ['environment', 'social', 'company'], 'scores': [0.9939545392990112, 0.453355073928833, 0.027347546070814133]}

template: In summary, the keyword is <mask>. res:{'sequence': 'green sustainability zero sustainable mitigation GREEN energy carbon proactive positive renewable yes Yes YES Green change Zero reduce renewables neutral action environmental low Carbon mitigating key adaptation target reduction Reduce local climate future relevant conservation good avoided active clean responsible environment progress efficiency balance focus there no Sustainable innovation C ', 'labels': ['environment', 'social', 'company'], 'scores': [0.9928809404373169, 0.628309965133667, 0.06896008551120758]}

template: <mask> is the keyword. res:{'sequence': 'Local Green Efficiency Affordable Sustainable Innovation Carbon Clean Solar Environment Energy Future Alternative Renew Smart This Community Effective Environmental Location Conservation local Reduction Responsibility GREEN Action Positive Zero Opportunity Climate Focus Reduce Specific Active Creative Simple Implementation Choice Savings Alternate Regional Yes Natural Low Solution Value Cool green Possible Technology ', 'labels': ['environment', 'social', 'company'], 'scores': [0.9926648139953613, 0.2627159059047699, 0.02313399687409401]}

#词性mask
template: In summary, the related word is <mask>. res:{'sequence': 'avoided green achieved used encouraged done recycled continued included provided implemented embraced renewable applied introduced sustainable supported adopted recommended linked growing connected GREEN promoted completed welcomed positive utilized advocated Green working increasing simplified rising reduced added related made addressed reused considered saved established key highlighted chosen checked mentioned reviewed delivered ', 'labels': ['environment', 'social', 'company'], 'scores': [0.99318927526474, 0.9521877765655518, 0.8532851338386536]}

template: In summary, the keyword is <mask>. res:{'sequence': 'green sustainable renewable local GREEN Green sustainability clean positive avoided environmental solar low simple recycling global circular neutral renewables natural good friendly environment ongoing zero Sustainable offset change responsible energy Renew key we future achieved innovative innovation relevant Local proactive YES conservation recycled new continued internal working emerging right efficient ', 'labels': ['environment', 'social', 'company'], 'scores': [0.9934384226799011, 0.4722353518009186, 0.16553758084774017]}

template: <mask> is the keyword. res:{'sequence': 'Green Renew Clean Energy Efficiency This Solar Local Environment Sustainable GREEN Innovation Future It Electricity Responsibility Alternative Focus Carbon That Action Sun Environmental Location Sunny Savings Light Natural Change Reduction Reduce Solution Goal Simple green Conservation What Sunshine Progress Choice Zero Wind Community Positive Friendly Cool Saving Awareness Solutions Success ', 'labels': ['environment', 'social', 'company'], 'scores': [0.9908861517906189, 0.8289745450019836, 0.05490453168749809]}
```
## 2022/11/16
1. 使用spacy进行dependency-parse，mask掉['nsubjpass','nsubj', 'dobj', 'amod']词性的词，并且非数字
  1. 分为两阶段，第一阶段完成mask，第二阶段模型直接读取mask后的文本
  2. 第一阶段先进行文本清洗，去掉了不可见字符，并合并空格，补空格，用spacy进行Mask，后将mask文本作为新的一列加入到数据集当中
  3. 模型读取数据的时候，直接将原文本和mask文本与运算，将非mask位置的label设置为-100
  4. roberta模型tokennizer分词的时候，一个单词可能被分为多个index，需要补上<mask>否则序列长度会不一样
```
('Interpublics', 'Directors', 'compound', 'NNS')
('Directors', 'elected', 'nsubjpass', 'NNS')
('are', 'elected', 'auxpass', 'VBP')
('elected', 'elected', 'ROOT', 'VBN')
('each', 'year', 'det', 'DT')
('year', 'elected', 'npadvmod', 'NN')
('by', 'elected', 'agent', 'IN')
('Interpublics', 'stockholders', 'compound', 'NNS')
('stockholders', 'by', 'pobj', 'NNS')
('at', 'elected', 'prep', 'IN')
('the', 'meeting', 'det', 'DT')
('annual', 'meeting', 'amod', 'JJ')
('meeting', 'at', 'pobj', 'NN')
('of', 'meeting', 'prep', 'IN')
('stockholders', 'of', 'pobj', 'NNS')
('.', 'elected', 'punct', '.')
('Interpublics', 'Committee', 'compound', 'NNS')
('Corporate', 'Governance', 'compound', 'NNP')
('Governance', 'Committee', 'compound', 'NNP')
('Committee', 'recommends', 'nsubj', 'NNP')
('recommends', 'recommends', 'ROOT', 'VBZ')
('nominees', 'recommends', 'dobj', 'NNS')
('to', 'recommends', 'prep', 'IN')
('the', 'Board', 'det', 'DT')
('Board', 'to', 'pobj', 'NNP')
('of', 'Board', 'prep', 'IN')
('Directors', 'of', 'pobj', 'NNPS')
(',', 'recommends', 'punct', ',')
('and', 'recommends', 'cc', 'CC')
('the', 'Board', 'det', 'DT')
('Board', 'proposes', 'nsubj', 'NNP')
('proposes', 'recommends', 'conj', 'VBZ')
('a', 'slate', 'det', 'DT')
('slate', 'proposes', 'dobj', 'NN')
('of', 'slate', 'prep', 'IN')
('nominees', 'of', 'pobj', 'NNS')
('to', 'proposes', 'prep', 'IN')
('the', 'stockholders', 'det', 'DT')
('stockholders', 'to', 'pobj', 'NNS')
('for', 'proposes', 'prep', 'IN')
('election', 'for', 'pobj', 'NN')

{(11, 'annual'), (30, 'proposes'), (21, 'nominees'), (33, 'slate'), (11, 'meeting'), (33, 'proposes'), (1, 'Directors'), (21, 'recommends'), (19, 'Committee'), (30, 'Board'), (19, 'recommends'), (1, 'elected')}

Interpublics Directors are elected each year by Interpublics stockholders at the annual meeting of stockholders. Interpublics Corporate Governance Committee recommends nominees to the Board of Directors, and the Board proposes a slate of nominees to the stockholders for election

 Interpublics <mask> are <mask> each year by Interpublics stockholders at the <mask> meeting of stockholders . Interpublics Corporate Governance <mask> <mask> <mask> to the Board of Directors , and the <mask> proposes a <mask> of nominees to the stockholders for election
```
1. 小样本训练：
--output_dir=/home/linzhisheng/esg/mlm/lzs_test --mask_stratagy=dynamic --data_path=/home/linzhisheng/esg/mlm/source_100w_mask --model_name_or_path=roberta-large --batch_size=16 --chunk_size=128 --training_size=100000 --do_train --do_eval
--output_dir=lzs_test --mask_stratagy=dynamic --data_path=/home/linzhisheng/esg/mlm/source_1w_mask --model_name_or_path=roberta-large --batch_size=16 --chunk_size=128 --training_size=10000 --do_train --do_eval

```

DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask', 'labels'],
        num_rows: 1283692
    })
})
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask', 'labels'],
        num_rows: 100000
    })
    test: Dataset({
        features: ['input_ids', 'attention_mask', 'labels'],
        num_rows: 10000
    })
})

***** Running Evaluation *****
  Num examples = 10000
  Batch size = 64
>>> Perplexity: 50.73

***** Running Evaluation *****
  Num examples = 10000
  Batch size = 64
>>> Perplexity: 8.51

# 输入
 "To contribute to climate change mitigation, we actively explore opportunities to support local renewable energy generation. Solar panels are installed at Hang Seng 113 to generate renewable energy."
 # 原始模型
 The sentence is about <mask>. :['sustainability', 'China', 'energy', 'electricity', 'solar', 'recycling', 'the', 'pollution', 'deforestation', 'waste', 'transparency', 'us</s>', 'emissions', 'green', 'cost', 'time', 'how', 'you', 'wind', 'water', 'power', 'money.', '2030', 'costs', 'why', 'sunset', 'construction', 'this', 'a', 'it', 'technology', 'cooling', 'Taiwan', 'them', 'science.', 'installation', 'change', 'renewables', 'Greenpeace', 'me', 'background', 'that', 'conservation', 'Asia', 'feasibility', 'Singapore', 'clouds']

 In summary, the related word is <mask>.:['sustainability', 'sustainable', 'green', 'solar', 'renewable', 'energy', 'Sustainable', 'zero', 'here', 'GREEN', 'mitigation', 'below', 'electricity', 'welcome', 'clean', 'wind', 'greenhouse', 'this', 'biomass', 'Solar', 'renewables', 'that', 'carbon', 'Green', 'Energy', 'environmental', 'positive', 'available', 'used', 'above', 'done:', 'unsustainable', 'simple', 'waste', 'important', 'power', 'effective', 'conservation', 'climate', 'water', 'sun', 'possible', 'achieved', 'abundant', 'efficiency', 'required', 'ongoing', 'significant', 'key']
# 100w数据上ramdom mask训练
 The sentence is about <mask>. :['solar', '2020', 'electricity', '2015', '2030', '2010', '2019', 'energy', '2014', '2012', '2011', 'sustainability', '2021', '2009', '2050', '2013', 'this', '2017', '2008', 'renewables', 'that', 'emissions', 'us', 'carbon', '2007', '2016', 'biomass', 'wind', '2018', 'efficiency', 'green', '2022', 'panels', '100', 'China', 'it', 'installation', '2', '10', '3', '2025', 'Singapore', 'zero', '2005', '15', 'them', '30', 'recycling', '1', '50']

 In summary, the related word is <mask>.:['green', 'energy', 'sustainable', 'sustainability', 'renewable', 'environment', 'environmental', 'GREEN', 'mitigation', 'Energy', 'Green', 'carbon', 'climate', 'relevant', 'positive', 'solar', 'electricity', 'support', 'zero', 'proactive', 'renewables', 'greenhouse', 'Environment', 'reduce', 'Sustainable', '', 'neutral', 'responsible', 'local', 'key', 'adaptation', 'Carbon', 'possible', 'enable', 'emission', 'appropriate', 'target', 'emissions', 'clean', 'innovative', 'important', 'global', 'ECO', 'essential', 'YES', 'active', 'friendly....', 'opportunities', 'mitigating']
# 10w数据上基于语法依存关系mask训练 
 The sentence is about <mask>. :['solar', 'electricity', 'energy', 'sustainability', 'green', 'wind', 'cooling', 'emissions', 'renewable', 'it', 'this', 'environmental', 'installation', 'Solar', 'China', 'conservation', 'Taiwan', 'efficiency', 'feasibility', 'local', 'us', 'Indonesia', 'investment', 'consumption', 'Singapore', 'recycling', 'water', 'power', 'location', 'panels', 'investments', 'environment', 'future', 'waste', 'resources', 'renewables', 'construction', 'potential', 'Kyoto', 'that', 'installations', 'procurement', 'biomass', 'saving', 'employees', 'climate', 'management', 'sustainable', 'carbon', 'buildings']

 In summary, the related word is <mask>.:['green', 'renewable', 'environmental', 'sustainable', 'sustainability', 'solar', 'energy', 'Green', 'environment', 'innovative', 'GREEN', 'proactive', 'Sustainable', 'mitigation', 'offset', 'responsible', 'promote', 'support', 'positive', 'local', 'clean', 'efficiency', 'conservation', 'indirect', 'relevant', 'efficient', 'alternative', 'inclusive', 'renewables', 'climate', 'friendly', 'Environment', 'Energy', 'voluntary', 'Environmental', 'Solar', 'neutral', 'electricity', 'complementary', 'integrated', 'innovation', 'global', 'related', 'potential', 'social', 'smart', 'welcome', 'resilient', 'adaptation', 'other']
- 根据语法依存关系进行划分，mask主谓宾和状语，有效去除了无意义的单词
```


## 2022/11/9
1. 在ESG数据集上基于random mask训练好的模型构造template，相关词提取top 50：
  1. Model：roberta-large
  2. Batch Size: 16 per GPU
  3. Training Size: 1000000
  4. Random mask 15%
```
input = "To mitigate the effects of global warming, we have been using eco-friendly refrigerants in our new air-conditioning systems."

The keyword is <mask>.:['efficient', 'efficiency', 'cool', 'cooling', 'green', 'effective', 'sustainable', 'smart', 'clean', 'new', 'affordable', 'inefficient', 'cheap', 'eco', 'good', 'cold', 'innovative', 'safe', 'sustainability', 'economical', 'plastic', 'natural', 'optimal', 'air', 'energy', 'water', 'right', 'recycling', 'choice', 'environmental', 'renewable', 'modern', 'climate', 'insulation', 'not', 'expensive', 'better', 'carbon', 'welcome', 'recycled', 'now', 'innovation', 'friendly', 'convenience', 'convenient', 'simple', 'moderation', 'artificial', 'inexpensive', 'organic']

 In summary, the related word is <mask>.:['cool', 'cooling', 'green', 'efficient', 'efficiency', 'cold', 'sustainable', 'effective', 'good', 'insulation', 'Cool', 'smart', 'welcome', 'carbon', 'energy', 'cooler', 'optimal', 'ozone', 'climate', 'awesome', 'chill', 'clean', 'sustainability', 'new', 'air', 'cheap', 'Celsius', 'plastic', 'zero', 'water', 'proactive', 'GREEN', 'excellent', 'innovation', 'nice', 'prudent', 'futuristic', 'additive', 'innovative', 'ICE', 'moderation', 'greenhouse', 'environmental', 'hydrogen', 'eco', 'affordable', 'this', 'hot', 'RED', 'conservation']

 In summary, the keyword word is <mask>.:['cool', 'green', 'cooling', 'good', 'sustainability', 'moderation', 'sustainable', 'efficiency', 'choice', 'effective', 'Cool', 'carbon', 'smart', 'cold', 'zero', 'energy', 'efficient', 'cooler', 'clean', 'climate', 'right', 'water', 'welcome', 'conservation', 'plastic', 'better', 'GREEN', 'air', 'innovation', 'control', 'simple', 'chill', 'insulation', 'optimal', 'Celsius', 'now', 'awesome', 'future', 'proactive', 'positive', 'this', 'flexibility', 'change', '…', 'wow', 'adaptation', 'enough', 'Green', 'new', 'heat']
 
 ['efficient', 'efficiency', 'cool', 'cooling', 'green', 'effective', 'sustainable', 'smart', 'clean', 'new', 'good', 'cold', 'sustainability', 'plastic', 'optimal', 'air', 'energy', 'water', 'climate', 'insulation', 'carbon', 'welcome', 'innovation', 'moderation']

 The keyword is <mask>.:['green', 'efficiency', 'efficient', 'ECO', 'sustainable', 'zero', 'eco', 'natural', 'energy', 'cool', 'GREEN', 'low', 'effective', 'water', 'carbon', 'clean', 'sustainability', 'Zero', 'conservation', 'safe', 'smart', 'Eco', 'Green', 'environmental', 'neutral', 'environment', 'innovative', 'recycled', 'cooling', 'NO', 'recycling', 'Cool', 'renewable', 'performance', 'climate', 'reduction', 'comfort', 'control', 'CO', 'economical', 'optimal', 'recycle', 'air', 'Smart', 'ecological', 'innovation', 'good', 'ozone', 'STAR', 'economy']

 In summary, the related word is <mask>.:['green', 'sustainable', 'cool', 'energy', 'efficient', 'cooling', 'environmental', 'eco', 'climate', 'environment', 'efficiency', 'sustainability', 'GREEN', 'innovative', 'zero', 'ECO', 'smart', 'new', 'Green', 'friendly', 'proactive', 'neutral', 'effective', 'carbon', 'good', 'innovation', 'greenhouse', 'excellent', 'clean', 'renewable', 'Cool', 'water', 'optimal', 'Eco', 'global', 'system', 'future', 'technology', 'warming', 'adaptation', 'safe', 'prudent', 'natural', 'conservation', 'air', 'Climate', 'emission', 'Environment', 'appropriate', 'welcome']

 In summary, the keyword word is <mask>.:['green', 'sustainability', 'sustainable', 'good', 'GREEN', 'environmental', 'Green', 'zero', 'energy', 'Eco', 'eco', 'excellent', 'environment', 'positive', 'yes', 'there', 'important', 'recycling', 'change', 'carbon', 'right', 'ECO', 'clean', 'responsible', 'key', 'YES', 'here', 'Yes', 'everywhere', 'progress', 'Sustainable', 'GOOD', 'done', 'effective', 'proactive', 'it', 'business', 'action', 'recycled', 'out', 'cool', 'climate', 'global', 'renewable', 'future', 'relevant', 'we', 'active', 'safe', 'solid']
 
 ['green', 'ECO', 'sustainable', 'zero', 'eco', 'energy', 'cool', 'GREEN', 'effective', 'carbon', 'clean', 'sustainability', 'safe', 'Eco', 'Green', 'environmental', 'environment', 'renewable', 'climate', 'good']
```
1. 使用nltk库获得词性标签，用于对形容词、名词进行mask    dbscan spacy    dependent tree 语法主体
  1. 形容词 'JJ'
  2. 名词 'NN'
  3. 动词 'VB'
"To mitigate the effects of global warming, we have been using eco-friendly refrigerants in our new air-conditioning systems."

[('To', 'TO'), ('mitigate', 'VB'), ('the', 'DT'), ('effects', 'NNS'), ('of', 'IN'), ('global', 'JJ'), ('warming', 'NN'), (',', ','), ('we', 'PRP'), ('have', 'VBP'), ('been', 'VBN'), ('using', 'VBG'), ('eco-friendly', 'JJ'), ('refrigerants', 'NNS'), ('in', 'IN'), ('our', 'PRP$'), ('new', 'JJ'), ('air-conditioning', 'JJ'), ('systems', 'NNS'), ('.', '.')]
3. 使用训练好的模型，结合KEYBERT进行关键词提取：

[图片]
```
"To mitigate the effects of global warming, we have been using eco-friendly refrigerants in our new air-conditioning systems."

[('conditioning', 0.9961), ('air', 0.9958), ('new', 0.9958), ('global', 0.9956), ('effects', 0.9955), ('using', 0.9954), ('friendly', 0.9952), ('eco', 0.9951), ('systems', 0.9948), ('mitigate', 0.9948), ('refrigerants', 0.9944), ('warming', 0.9943)]

[('conditioning', 0.9929), ('new', 0.9917), ('using', 0.9916), ('friendly', 0.9914), ('warming', 0.991), ('systems', 0.9909), ('global', 0.9902), ('refrigerants', 0.9898), ('eco', 0.9898), ('mitigate', 0.9895), ('effects', 0.9894), ('air', 0.9886)]
```
4. MDERank: A Masked Document Embedding Rank Approach for 
Unsupervised Keyphrase Extraction
https://aclanthology.org/2022.findings-acl.34.pdf
[图片]


# Train
python main2.py --output_dir lzs_test --mask_stratagy random --data_path source_1w --model_name_or_path roberta-large --batch_size 16 --chunk_size 128 --training_size 100 --do_train --do_eval


python main2.py --output_dir /home/linzhisheng/esg/mlm/200w_mask --mask_stratagy dynamic --data_path /home/linzhisheng/esg/mlm/source_200w_mask --model_name_or_path roberta-large --batch_size 16 --chunk_size 128 --training_size 100 --do_train --do_eval

python main2.py --output_dir /root/ESG/mlm/200w_mask --mask_stratagy dynamic --data_path /root/ESG/mlm/source_mask_80% --model_name_or_path roberta-large --batch_size 16 --chunk_size 128 --training_size 2000000 --do_train --load_cache

nohup python -u main2.py --output_dir /root/ESG/mlm/200w_mask --mask_stratagy dynamic --data_path /home/linzhisheng/esg/mlm/source_mask_80% --model_name_or_path roberta-large --batch_size 64 --chunk_size 128 --training_size 100000 --do_train --do_eval --load_cache > outout.txt &


nohup python -u main2.py --output_dir /root/ESG/mlm/esg-100w-model --mask_stratagy random --data_path /root/ESG/mlm/source_mask_80% --model_name_or_path roberta-large --batch_size 60 --chunk_size 128 --training_size 1000000 --do_train --do_eval --load_cache_dir /root/ESG/mlm/lm_datasets_100w --tokenizer_path /root/ESG/mlm/roberta-esg-tokenizer > outout.txt &