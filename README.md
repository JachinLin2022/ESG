# Update

## Reference
1. NLP预训练中的mask方式总结 https://zhuanlan.zhihu.com/p/434672623
2. MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction https://aclanthology.org/2022.findings-acl.34.pdf
3. zero-shot learning https://joeddav.github.io/blog/2020/05/29/ZSL.html
4. FinBERT: Financial Sentiment Analysis with Pre-trained Language Models https://arxiv.org/pdf/1908.10063.pdf
5. SCI BERT: A Pretrained Language Model for Scientific Text  https://arxiv.org/pdf/1903.10676.pdf
6. Domain-Specific BERT Models https://mccormickml.com/2020/06/22/domain-specific-bert-tutorial/#3-comparing-scibert-and-bert
## 2022/12/22
1. 基于词性mask，mask掉两边单词
2. 选取第三方模型bert-base-nli-mean-tokens提取word embedding，计算cos相似度。MRR是一个国际上通用的对搜索算法进行评价的机制，即第一个结果匹配，分数为1，第二个匹配分数为0.5，第n个匹配分数为1/n，如果没有匹配的句子分数为0。最终的分数为所有得分之和。
3. 词表评估实验结果
  + 基于概率，选取概率最高的前top-n个单词计算相似度，求和取平均
    （1）引入权重，MRR
    （2）不引入权重，词表提取部分与下游评价部分分离
  + 基于词频，选取词频最高的前top-n个单词计算相似度，求和取平均


## 2022/12/15
1. 后续方案讨论
  + 关键词信息问答，如输入emisstion ，模型输出排放量相关的数字，需要使用GPT或者T5等基于生成式任务的预训练模型并微调
  + 使用词表进行分类，计算词表中单词与该句子标签的距离，使用第三方模型得到word embedding计算距离
1. 完成基于词性mask的模型训练
  + 基于词性mask 80%，20%不变
  + 基于词性mask 80%，10%不变，10%随机替换
1. 模型对比
  + 原始模型
  + random-mask
  + fin-bert
  + 基于词性mask 80%，20%不变
  + 基于词性mask 80%，10%不变，10%随机替换

## 2022/12/8
1. 完成全量数据random mask训练
2. Template对比选择
```
The Keyword is <mask>.:['1', '2', '3', 'energy', 'energy', '2030', 'a', '5', 'climate', 'a', 'zero', '7', '4', '2025', 'renewable', 'renewable', 'progress', '2020', '17', '15', 'goal', '6', 'ii', '10', 'change2', '100', '2021', 'cdp', 'mitigation', 'c', 'co', 'below', 'b', '"', 'environment', '9', 'change', 'sustainability', 'i', '2050', '41', 'that', 'green', 'sdg', 'more', 'included', 'carbonneutral', 'reduce', 'electricity', 'no.', 'yes', 'environmental', 'target', 'priority', 'the', 'zero', 'better', 'emissions', 'one', 'efficiency', 'decarbonisation', '13', 'first', 'carbon', '8', 'planet', 'e', 'above:', 'offset', 'biodiversity', 'further', 'target', 'un', 'committed', 'reduced', 'future', 'transition', 'wind', 'ecoefficiency', '12', 'emissions', 'global', 'opportunities', 'climate', 'ghg', '11', 'the', 'three', 'smart', 'this', 'reduction', 'two', '163', 'however', 'i', 'renewables']
The keyword is <mask>.:['energy', 'mitigation', 'transition', 'decarbonisation', 'opportunities', 'efficiency', 'electricity', 'environment', 'carbon', 'renewable', 'climate', 'change', 'biodiversity', 'renewables', 'demand', '2030', 'action', 'adaptation', 'emissions', 'society', 'ecoefficiency', 'sustainability', '2025', 'warming', 'nature', 'priority', 'resilience', 'energysaving', 'prevention', 'economy', 'environmental', 'risk', '2020', 'future', 'temperature', 'carbonneutral', '2050', 'growth', 'that', 'climate', 'power', 'reduction', 'savings', 'transparency', '1', 'target', 'progress', 'inclusion', 'support', 'energy', 'innovation', 'more', 'sdg', 'deforestation', 'saving', 'wind', 'how', 'improvement', 'offsets', 'business', 'infrastructure', 'first', 'resources', 'compliance', 'energyefficiency', 'poverty', 'this', 'strategy', 'lowcarbon', 'transport', 'potential', 'customers', '2021', 'cdp', 'cdm', 'global', 'disaster', 'consumption', 'education', 'neutrality', 'regulation', 'diversification', 'costs', 'impact', 'variation', 'biogas', 'development', 'renewable', 'planet', 'opportunity', 'access', 'decarbonization', '2023', '2', 'cyberattacks', 'sustainability', 'financing', 'commitment', 'offsetting', 'clean']
In summary, the related word is <mask>.:['1', 'i', 'a', '3', '2', 'below', 'b', 'a', 'used', 'e', 'c', 'ii', 'en', 'i', 'ghg', '"', 'cdp', 'b', 'co', '4', '7', 'renewable', 'carbon', 'energy', 'l', 'g', 'attached', 'included', 'that', '5:.', 't', 'c', 'ace', 'the', '*', 'energy', 'see', 'd', '15', '20252', '17', '3021', 'gj', 'e', 'electricity', 'na', '10', 'renewable', 'k', 'mm', 'no', 'gri', 'kpi', 'f', 'v', 'follows', 'p', 'reduced', 'green', 'p', '6', '100', 'yes', 're', 'r', 'g', 't', 'more', 'www', 'zeroa', 'otherwise', 'no', 'zerorefer', '9', '53', 'subject', 'above', '2030', 'limited', 'provided', 'formosa', 'voc', 'applied', 'covered', 'viz', 'wri', 'rohs', 'kyoto', 'go', 'either', 'sasb', '>', 'iso', '8', 'only']
In summary, the keyword is <mask>.:['1', 'energy', 'mitigation', '2', '3:', 'electricity', 'that', 'emissions', '2030', 'below', 'carbon', '2025', '5', 'decarbonisation', '2050', 'change', '2020', 'a', '4', 'zero', 'transition', 'ecoefficiency', 'ii', 'efficiency', 'renewable', '7', 'biodiversity', 'zero', 'cdp', 'sustainability', 'risk', 'three', 'climaterelated', 'opportunities', 'c', 'environment', '2021', 'climate', 'carbonneutral', 'climate', 'co', 'renewables', 'energy', 'a', '62', '17', 'target', '"', 'priority', '15', '9', 'ghg', 'future', 'goal', 'b', 'i', 'the', '10', 'this', 'progress', 'adaptation', 'e', 'above', '41', 'b', 'to', 'reduce', 'more', '100', 'included', 'global', 'lower', 'used', 'c', 'sdg', 'reduction', '2023', 'here', 'following.', 'however', 'how', 'offsets', 'ghgs', 'possible', 'action', '2018', 'two', '2040', '2016', 'business', '2012', 'environmental', 'prevention', 'six', 'kpi', '16', 'demand']
<mask> is the related word. :['which', 'and)', 'this', 'it', 'this', 'emissions', 'that23.', 'energy', 'site', 'energy', 'footprint', 'as),', 'environment', 'group', 'change', '2', 'world', 'it', 'or', 'company', 'change', '2020', 'use', 'ghg', '4', 'therea', 'name', 'warming4', '1x', 'the', 'scope', 'consumptionee,', 'country', 'scope', 'data', 'chain', 'electricity', 'below', '*5', 'we', 'product1', 'areab', 'japan', '3', 'that', 'environment', 'year', 'also', 'euc', '2017', 'china', 'example', 'canada', 'atmosphere', '2019', 'sites', 'a', '5q', 'gas', 'system', 'e', 'report', 'project.', 'code', 'water', 'kyoto', 'emission', '2015', 'one', '', 'project', 'sustainabilityn', 'america")', '2018', 'organization', 'what', 'category', 'climate', 'a']
<mask> is the keyphrase. :['energy', 'change', 'economy', 'warming', 'environment', 'business', 'generation', 'emissions', 'power,', 'development', 'consumption', 'climate', 'use', 'society', 'it', 'future', 'world', 'electricity', 'operations', 'resources', 'industry', 'production', 'gas', 'source', 'atmosphere', 'company', 'customers', 'efficiency', 'footprint', 'transition', 'growth', 'infrastructure', 'that', 'weather', 'supply', 'chain', 'activity', 'nature', 'and', 'biodiversity', 'demand', 'operation', 'water', 'which', 'area', 'technology', 'sector', 'pollution', 'site', 'disaster', 'country', 'temperature', 'building', 'time', 'carbon', 'product', 'system', 'wind2', 'this', 'group', 'life', 'fuels', 'energy', 'impact', 'region', 'fuel', 'population', 'transportation', 'opportunities', 'opportunity', 'technologies', 'transport', 'emission', 'conservation', 'community', 'buildings', 'gases', 'sustainability', 'market', 'travel', 'saving)', 'ecosystem', 'sources', 'mitigation', 'reduction', 'grid', 'products', 'customer', 'policy', 'process', 'activities', 'aviation', 'solution', 'innovation', 'communities', 'effect', 'equipment']
<mask> is the keyword. :['energy', 'change', 'warming', 'environment', 'economy', 'emissions', 'society,', 'consumption', 'development', 'it', 'business', 'climate', 'power', 'generation', 'future', 'use', 'that', 'electricity', 'transition', 'which', 'operations', 'world', 'growth', 'efficiency', 'atmosphere', 'resources', 'customers', 'production', 'nature', 'biodiversity', 'and', 'footprint', 'water', 'pollution', 'mitigation', 'reduction', 'temperature', 'company', 'activity', 'energy', 'conservation', 'disaster', 'industry', 'sustainability', 'saving', 'carbon', 'supply', 'technology', 'demand', 'gas', 'source', 'chain', 'wind', 'this', 'site', 'operation', 'building', 'opportunities', 'opportunity', 'infrastructure', 'life', 'policy', 'product', 'area', 'action', 'weather', 'time', '2020"', 'impact2', 'innovation', 'system', 'california', 'buildings', 'protection', 'travel', 'management', 'products', 'fuel', 'this', 'emission', 'technologies', 'activities', 'regulation.', 'solutions', 'sector', 'project', 'community', 'solution', 'country', 'recovery', 'legislation', 'transport', 'risk)', 'goal', '2050']
In summary, <mask> is the keyphrase. :['energy', 'electricity', 'production', 'power', 'technology', 'renewables', 'climate', 'efficiency', 'transport', 'demand', 'sustainability', 'environment', 'generation', 'warming', 'consumption', 'carbon', 'nature', 'emissions', 'energy', 'biodiversity', 'water', 'it', 'renewable', 'energysaving', 'wind', 'growth', 'business', 'innovation', 'transportation', 'biogas', 'aviation', 'infrastructure', 'resources', 'gas', 'fuel', 'temperature', 'mobility', 'society', 'travel', 'economy', 'procurement', 'agriculture', 'development', 'emission', 'economic', 'this', 'quality', 'solar', 'buildings', 'utilities', 'heating', 'utility', 'operations', 'hydropower', 'safety', 'logistics', 'resilience', 'pollution', 'comfort', 'energyefficiency', 'operation', 'revenue', 'fuels', 'weather', 'manufacturing', 'biofuels', 'customers', 'lighting', 'kyoto', 'environmental', 'decarbonization', 'industry', 'energies', 'regulation', 'education', 'steam', 'savings', 'transition', 'electricity', 'cogeneration', 'usage', 'cement', 'use', 'recycling', 'investment', 'ghg', 'our', 'security', 'coal', 'productivity', '2020', 'electrification', 'supply', 'heat', 'china', 'cost', 'change', 'it', 'management', 'housing']
In summary, <mask> is the keyword. :['energy', 'warming', 'climate', 'mitigation', 'transition', 'environment', 'decarbonisation', 'efficiency', 'sustainability', 'resilience', 'society', 'nature', 'change', 'adaptation', 'carbon', 'emissions', 'biodiversity', 'electricity', 'action', 'climate', 'energy', 'energysaving', 'temperature', 'deforestation', 'safety', 'renewables', 'demand', 'prevention', 'poverty', 'renewable', '2020', 'it', 'offsetting', 'power', 'ecoefficiency', 'growth', 'reduction', 'this', 'future', '2050', '2030', 'regulation', 'decarbonization', 'water', 'environmental', 'consumption', 'technology', 'flooding', 'pollution', 'wind', 'progress', 'planet', 'target', 'ghg', 'emission', 'business', 'flood', 'kyoto', 'energyefficiency', 'efficiency', 'compliance', 'diversification', 'innovation', 'risk', 'neutrality', 'savings', 'economy', 'cdm', 'ghgs', 'offsets', 'science', 'awareness', '2025', 'saving', 'uncertainty', 'variation', 'urbanization', 'opportunities', 'disaster', 'atmosphere', 'inclusion', 'cdp', 'sustainability', 'carbonneutral', 'weather', 'priority', 'education', 'csr', 'development', 'sdg', 'decarbon', 'transparency', 'travel', 'biodiversity', 'security', 'zero', 'circularity', 'comfort', 'goal', 'aviation']
In summary, <mask> is the related word. :['this', 'energy', 'energy', 'this', 'it', 'climate', 'the', 'that', 'cdp', 'renewable', 'ghg', 'climate', 'carbon', 'co', '"', 'carbon', 'our', 'emissions', 'we', '2020', 'change', 'ghgs', '2050', 'a', 'planet', 'environmental', 'there', 'these', '*', 'c', 'it', 'no', 'hydro', '2', 'what', 'what', 'kyoto', '2021', 'csr', 'cdm', 'sustainability', '3021', '2025', 'energy', 'i', 'electricity', 'zero', 'which', 't', 'renewables', 'nox', 'sustainability', '2030', 'warming', '1', 'ace', 'change', 'target', 'e', 'waste', 'that', 'kpi', 'there', 'a2', 'these', 'target', 'environment', 'environmental', 'solar', 'e', 'goal', 'green', 'zero', 'i', 'sasb', 'b', 'voc', 'cem', '3', 'it', '15', 'future', 'led', 'decarbonisation', 'biogas', 'emissions', 'smart', 'three', '2016', 'tcfd', '2015', 'esg', 'gri', '2013', 'methane', 'scope', 'its', '100', 'water']


'<mask> is the keyphrase. ',
'<mask> is the keyword. ',
'In summary, <mask> is the keyphrase. ',
'In summary, <mask> is the keyword. '
```
3. 对一千条文本，单个模板，提取关键词表，并集后统计词频，显示词云图


|  top-k   | random-mask  | 原始模型 |
|  ----  | ----  | ---- |
| 100  | [('management', 1237), ('sustainability', 1059), ('safety', 987), ('diversity', 825), ('what', 818), ('compliance', 793), ('there', 786), ('information', 785), ('performance', 761), ('these', 739), ('quality', 695), ('one', 660), ('csr', 646), ('governance', 626), ('inclusion', 602), ('directors', 585), ('each', 583), ('responsibility', 566), ('people', 524), ('data', 523), ('we', 522), ('employees', 510), ('he', 506), ('integrity', 502), ('innovation', 501), ('such', 485), ('2018', 482), ('technology', 478), ('environment', 466), ('2020', 463), ('leadership', 460), ('below', 457), ('ethics', 455), ('business', 441), ('our', 441), ('esg', 440), ('gender', 438), ('training', 437), ('transparency', 424), ('energy', 421), ('shareholders', 409), ('reference', 390), ('education', 387), ('2015', 377), ('reporting', 374), ('they', 355), ('society', 355), ('women', 354), ('development', 349), ('2013', 349), ('health', 347), ('strategy', 345), ('board', 338), ('engagement', 338), ('environmental', 336), ('communication', 334), ('excellence', 333), ('collaboration', 331), ('water', 331), ('independence', 321), ('following', 319), ('above', 317), ('remuneration', 304), ('disclosure', 300), ('2017', 299), ('executive', 292), ('director', 287), ('risk', 275), ('change', 269), ('hr', 268), ('as', 268), ('biodiversity', 265), ('fairness', 264), ('service', 262), ('chairman', 256), ('success', 251), ('security', 250), ('growth', 247), ('here', 245), ('you', 243), ('compensation', 238), ('2008', 238), ('staff', 234), ('reelection', 232), ('efficiency', 231), ('2016', 231), ('work', 230), ('independent', 228), ('3', 227), ('climate', 227), ('credibility', 226), ('ir', 225), ('productivity', 223), ('production', 222), ('ceo', 217), ('everyone', 217), ('recruitment', 213), ('control', 213), ('a', 209), ('example', 205), ('materiality', 204), ('equity', 204), ('succession', 198), ('2009', 198), ('prevention', 197), ('company', 197), ('ehs', 193), ('2012', 193), ('electricity', 190), ('awareness', 189)] |[('management', 1237), ('sustainability', 1059), ('safety', 987), ('diversity', 825), ('what', 818), ('compliance', 793), ('there', 786), ('information', 785), ('performance', 761), ('these', 739), ('quality', 695), ('one', 660), ('csr', 646), ('governance', 626), ('inclusion', 602), ('directors', 585), ('each', 583), ('responsibility', 566), ('people', 524), ('data', 523), ('we', 522), ('employees', 510), ('he', 506), ('integrity', 502), ('innovation', 501), ('such', 485), ('2018', 482), ('technology', 478), ('environment', 466), ('2020', 463), ('leadership', 460), ('below', 457), ('ethics', 455), ('business', 441), ('our', 441), ('esg', 440), ('gender', 438), ('training', 437), ('transparency', 424), ('energy', 421), ('shareholders', 409), ('reference', 390), ('education', 387), ('2015', 377), ('reporting', 374), ('they', 355), ('society', 355), ('women', 354), ('development', 349), ('2013', 349), ('health', 347), ('strategy', 345), ('board', 338), ('engagement', 338), ('environmental', 336), ('communication', 334), ('excellence', 333), ('collaboration', 331), ('water', 331), ('independence', 321), ('following', 319), ('above', 317), ('remuneration', 304), ('disclosure', 300), ('2017', 299), ('executive', 292), ('director', 287), ('risk', 275), ('change', 269), ('hr', 268), ('as', 268), ('biodiversity', 265), ('fairness', 264), ('service', 262), ('chairman', 256), ('success', 251), ('security', 250), ('growth', 247), ('here', 245), ('you', 243), ('compensation', 238), ('2008', 238), ('staff', 234), ('reelection', 232), ('efficiency', 231), ('2016', 231), ('work', 230), ('independent', 228), ('3', 227), ('climate', 227), ('credibility', 226), ('ir', 225), ('productivity', 223), ('production', 222), ('ceo', 217), ('everyone', 217), ('recruitment', 213), ('control', 213), ('a', 209), ('example', 205), ('materiality', 204), ('equity', 204), ('succession', 198), ('2009', 198), ('prevention', 197), ('company', 197), ('ehs', 193), ('2012', 193), ('electricity', 190), ('awareness', 189)]|
| 1000  | [('sustainability', 1960), ('management', 1893), ('company', 1788), ('diversity', 1692), ('inclusion', 1654), ('information', 1646), ('these', 1636), ('safety', 1623), ('compliance', 1525), ('responsibility', 1511), ('governance', 1494), ('what', 1489), ('our', 1481), ('data', 1461), ('performance', 1434), ('there', 1408), ('corporate', 1377), ('employees', 1375), ('people', 1353), ('first', 1349), ('a', 1344), ('quality', 1337), ('ethics', 1295), ('all', 1286), ('one', 1280), ('women', 1261), ('innovation', 1250), ('technology', 1225), ('gender', 1211), ('board', 1209), ('risk', 1206), ('fairness', 1203), ('group', 1195), ('integrity', 1181), ('business', 1173), ('transparency', 1171), ('no', 1151), ('excellence', 1147), ('shareholders', 1144), ('change', 1142), ('i', 1140), ('engagement', 1137), ('energy', 1133), ('leadership', 1129), ('environment', 1128), ('they', 1126), ('none', 1123), ('results', 1119), ('target', 1114), ('he', 1109), ('independence', 1101), ('others', 1093), ('health', 1089), ('directors', 1083), ('training', 1076), ('scope', 1072), ('executive', 1066), ('progress', 1066), ('operations', 1062), ('transformation', 1058), ('c', 1046), ('employment', 1038), ('audit', 1030), ('action', 1027), ('remuneration', 1026), ('employee', 1026), ('security', 1019), ('success', 1010), ('growth', 1010), ('ownership', 1007), ('us', 1004), ('each', 1000), ('reference', 1000), ('however', 997), ('such', 993), ('equity', 990), ('', 988), ('environmental', 988), ('development', 983), ('education', 978), ('society', 978), ('compensation', 977), ('example', 974), ('reporting', 968), ('evaluation', 965), ('work', 964), ('its', 964), ('annual', 963)] |[('here', 2744), ('below', 2128), ('data', 1757), ('what', 1750), ('there', 1682), ('information', 1674), ('management', 1669), ('transparency', 1655), ('following', 1651), ('above', 1648), ('focus', 1604), ('risk', 1583), ('price', 1575), ('as', 1567), ('technology', 1566), ('time', 1527), ('disclosure', 1515), ('compliance', 1508), ('performance', 1499), ('cost', 1475), ('context', 1464), ('change', 1461), ('one', 1459), ('access', 1436), ('value', 1421), ('summary', 1407), ('communication', 1406), ('overview', 1403), ('outlook', 1389), ('balance', 1385), ('experience', 1382), ('compensation', 1358), ('education', 1358), ('priority', 1349), ('diversity', 1325), ('policy', 1307), ('trust', 1301), ('security', 1287), ('identity', 1266), ('control', 1263), ('equity', 1263), ('strategy', 1257), ('background', 1252), ('next', 1242), ('why', 1241), ('participation', 1240), ('exposure', 1240), ('how', 1238), ('safety', 1236), ('history', 1233), ('research', 1232), ('leadership', 1227), ('key', 1222), ('evolution', 1210), ('a', 1208), ('success', 1205), ('first', 1199), ('business', 1195), ('content', 1192), ('opportunity', 1182), ('responsibility', 1168), ('money', 1168), ('respect', 1162), ('he', 1158), ('analysis', 1136), ('employment', 1133), ('care', 1133), ('x', 1128), ('operation', 1125), ('software', 1121), ('accountability', 1114), ('discipline', 1109), ('volume', 1108), ('support', 1105), ('impact', 1104), ('action', 1100), ('income', 1099), ('quality', 1090), ('scope', 1090), ('progress', 1084), ('competition', 1080), ('funding', 1076), ('now', 1072), ('reporting', 1061), ('implementation', 1053), ('availability', 1052), ('if', 1049), ('today', 1046), ('selection', 1045), ('these', 1044), ('regulation', 1032), ('revenue', 1032), ('independence', 1029), ('identification', 1019), ('healthcare', 1018)]|
| 10000  | [('sustainability', 2744), ('all', 2737), ('first', 2714), ('one', 2713), ('company', 2701), ('inclusion', 2621), ('next', 2620), ('us', 2610), ('no', 2605), ('total', 2567), ('employees', 2558), ('diversity', 2557), ('target', 2545), ('annual', 2487), ('operations', 2457), ('others', 2437), ('lead', 2432), ('life', 2432), ('integrity', 2406), ('accountability', 2395), ('corporate', 2368), ('shareholders', 2363), ('an', 2355), ('performance', 2340), ('employment', 2334), ('group', 2332), ('contents', 2331), ('social', 2327), ('you', 2317), ('customers', 2309), ('anticorruption', 2308), ('progress', 2307), ('reporting', 2290), ('energy', 2290), ('quality', 2289), ('association', 2280), ('people', 2277), ('business', 2276), ('purpose', 2270), ('scope', 2267), ('equality', 2267), ('information', 2264), ('equity', 2264), ('society', 2263), ('voting', 2254), ('wellbeing', 2248), ('ethics', 2245), ('responsibility', 2244), ('safety', 2243), ('legal', 2242), ('independence', 2240), ('public', 2239), ('data', 2231), ('general', 2224), ('objective', 2221), ('re', 2218), ('our', 2214), ('yes', 2213), ('report', 2208), ('investors', 2207), ('global', 2203), ('elearning', 2202), ('women', 2201), ('management', 2198), ('area', 2196), ('core', 2195), ('nonexecutive', 2194), ('development', 2193), ('conduct', 2190), ('impact', 2180), ('audit', 2164), ('man', 2157), ('compliance', 2156), ('as', 2156), ('focus', 2138), ('vision', 2137)] |[('here', 2955), ('as', 2935), ('there', 2916), ('care', 2902), ('he', 2887), ('next', 2883), ('what', 2869), ('top', 2844), ('now', 2842), ('red', 2828), ('rest', 2824), ('time', 2802), ('how', 2799), ('so', 2794), ('she', 2705), ('data', 2704), ('today', 2672), ('step', 2668), ('below', 2652), ('link', 2618), ('one', 2616), ('up', 2569), ('these', 2547), ('life', 2533), ('first', 2526), ('gold', 2521), ('why', 2501), ('power', 2481), ('art', 2473), ('green', 2450), ('select', 2442), ('map', 2429), ('if', 2409), ('ed', 2403), ('home', 2397), ('see', 2395), ('act', 2390), ('net', 2386), ('live', 2372), ('control', 2340), ('action', 2328), ('read', 2315), ('update', 2300), ('th', 2296), ('cap', 2291), ('star', 2277), ('key', 2270), ('no', 2256), ('news', 2256), ('is', 2248), ('name', 2240), ('list', 2240), ('may', 2229), ('sum', 2228), ('yes', 2227), ('in', 2220), ('cat', 2203), ('support', 2201), ('go', 2200), ('code', 2199), ('me', 2181), ('re', 2176), ('us', 2174), ('man', 2170), ('out', 2164), ('dis', 2162), ('at', 2153), ('fire', 2150), ('stop', 2149), ('new', 2146), ('last', 2141), ('path', 2135), ('note', 2135), ('set', 2129), ('nor', 2127), ('high', 2127), ('em', 2127), ('love', 2114), ('case', 2106), ('id', 2103), ('war', 2093), ('far', 2082), ('hit', 2076), ('plan', 2071), ('add', 2063), ('par', 2061), ('pay', 2055), ('information', 2052), ('work', 2039), ('space', 2039), ('mark', 2033), ('video', 2032), ('open', 2031), ('down', 2020)]|

4. 对单条文本，多个模板，提取词表统计词频，显示词云图
5. 完成部分基于词性Mask的训练
+ Mask 所有词性单词
[图片]
+ Mask 80%单词
+ Mask 非ROOT类单词
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
1. 基于ESG语料训练roberta tokenizer
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


nohup python -u main2.py --output_dir /root/ESG/mlm/esg-roberta-model --mask_stratagy random --data_path /root/ESG/mlm/source_mask_80% --model_name_or_path roberta-large --batch_size 60 --chunk_size 128 --training_size 1000000 --do_train --do_eval --load_cache_dir /root/ESG/mlm/esg_datasets_all --tokenizer_path /root/ESG/mlm/roberta-esg-tokenizer > outout.txt &


<!-- own use -->
python  main2.py --output_dir /home/linzhisheng/esg/mlm/esg-roberta-model --mask_stratagy random --data_path /home/linzhisheng/esg/mlm/source_mask_80% --model_name_or_path roberta-large --batch_size 16 --chunk_size 128 --training_size 1000000 --do_train --do_eval --load_cache_dir /home/linzhisheng/esg/mlm/esg_datasets_all --tokenizer_path /home/linzhisheng/esg/mlm/roberta-esg-tokenizer