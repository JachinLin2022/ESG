import json
import pandas as pd
# extract info from source
def format_source():
    source = pd.read_csv('/usr/yubomai/esg/sources/all.csv',error_bad_lines=False)
    SourceAbstract = source[source['Path'] == 'Sources/SourceAbstract'][['ObjectId','Value']]
    SourcePublisher= source[source['Path'] == 'Sources/SourcePublisher'][['ObjectId','Value']]
    SourceUrl= source[source['Path'] == 'Sources/SourceUrl'][['ObjectId','Value']]
    SourceTitle = source[source['Path'] == 'Sources/SourceTitle'][['ObjectId','Value']]
    SourceAbstract = pd.merge(SourceAbstract,SourcePublisher,how="left", on="ObjectId")
    SourceAbstract = pd.merge(SourceAbstract,SourceUrl,how="left", on="ObjectId")
    SourceAbstract = pd.merge(SourceAbstract,SourceTitle,how="left", on="ObjectId")
    SourceAbstract.columns = ['ObjectId', 'Abstract', 'Publisher', 'Url', 'Title']

    print(SourceAbstract)
    SourceAbstract.to_csv('source_title.csv',index=False)

# def format_schemeA():
#     schemeA = pd.read_csv('/usr/yubomai/esg/schemeA/schemeA.csv',error_bad_lines=False,keep_default_na=False)
#     schemeA_not_null = schemeA[schemeA['Value'] != 'null']
#     schemeA_not_null.to_csv('schemeA.csv',index=False)
#     print(schemeA_not_null)

def format_schemeB():
    schemeB = pd.read_csv('/usr/yubomai/esg/schemeB/social',error_bad_lines=False,keep_default_na=False)
    schemeB_not_null = schemeB[schemeB['Value'] != 'null']
    schemeB_not_null.to_csv('social.csv',index=False)
    print(schemeB)

def esg_format():
    source = pd.read_csv('source.csv',error_bad_lines=False,keep_default_na=False,nrows=100)
    env = pd.read_csv('gov.csv',error_bad_lines=False,keep_default_na=False)
    env = env[env['Path'].str.find('DataPoints')>=0]
    # env['ObjectId'] = env['StatementDetails/OrganizationId']
    env = env[['Path', 'Value']]
    source['ObjectId'] = source['ObjectId'].astype(str)
    # print(source['ObjectId'].dtype)
    # pd.to_numeric(env['Value'])


    print(env['Value'].dtype)
    source = pd.merge(source,env,how="left", left_on="ObjectId", right_on='Value')

    source.to_csv('res1.csv',index=False)



    # print(source.head(10))
    # print(env)

def get_res(source, file):
    t = pd.read_csv(file,error_bad_lines=False,keep_default_na=False)
    sourceid = t[t['Path'].str.find('SourceId')>=0]
    ValueScore = t[t['Path'].str.find('ValueScore')>=0]
    ValueScore = ValueScore[['ObjectId', 'Path', 'Value']]
    sourceid = sourceid[['ObjectId', 'Path', 'Value']]
    sourceid['extra'] = sourceid['Path'].apply(lambda st: st[0:st.find('/Sources')])
    # t = t[t['Path'].str.find('SourceId') >= 0]
    ValueScore['extra'] = ValueScore['Path'].apply(lambda st: st[0:st.find('/ValueScore')])
    join = pd.merge(ValueScore,sourceid,how="left", on=['ObjectId','extra'])
    join = join.drop(labels=['extra','Path_y'], axis=1)
    join = join.rename(columns={'Value_y':'SourceId','Path_x':'Path','Value_x':'ValueScore'})
    join['SourceId'] = join['SourceId'].fillna('null')
    join = join[join['SourceId'] != 'null']    

    join = pd.merge(join,source,how="left", left_on='SourceId',right_on='ObjectId')

    join = join.drop(labels=['ObjectId_y'], axis=1)
    return join
def test():
    source = pd.read_csv('source_title.csv',error_bad_lines=False,keep_default_na=False)
    print('read souce done')
    source['ObjectId'] = source['ObjectId'].astype(str)
    join_social = get_res(source,'social.csv')
    join_env = get_res(source,'env.csv')
    join_gov = get_res(source,'gov.csv')
    res = pd.concat([join_social,join_env,join_gov],axis=0)
    print('start to dump res')
    res[['Publisher','Title']].to_csv('title.csv', index = False)
    # res.to_csv('res.csv',index=False)

def filter(soucer, list):
    for i in list:
        soucer = soucer[soucer['Publisher'].str.find(i) < 0]
    return soucer
def classify():
    Title = pd.read_csv('title.csv')
    Title['Title'] = Title['Title'].str.lower()
    Title['Publisher'] = Title['Publisher'].str.lower()
    # Title.drop_duplicates(inplace=True)
    # filter(Title, ['ltd', 'cor', 'company', 'inc', 'plc']).to_csv('t.csv',index=False)
    Title_with_report = Title[Title['Title'].str.find('report') >= 0]
    Title_with_annual_report = Title_with_report[Title_with_report['Title'].str.find('annual') >= 0]

    print('total num is {0}'.format(Title.shape[0]))
    print('report num is {}, annual have {}'.format(Title_with_report.shape[0], Title_with_annual_report.shape[0]))
    
    # Title_with_report.to_csv('t.csv',index=False)
    
    Title_without_report = Title[Title['Title'].str.find('report') < 0]
    Title_with_code = Title_without_report[Title_without_report['Title'].str.find('code') >= 0]
    Title_with_policy = Title_without_report[Title_without_report['Title'].str.find('policy') >= 0]
    Title_with_ethics = Title_without_report[Title_without_report['Title'].str.find('ethic') >= 0]
    t = Title_without_report[(Title_without_report['Title'].str.find('sustai') >= 0) | (Title_without_report['Title'].str.find('gov') >= 0) | (Title_without_report['Title'].str.find('env') >= 0) | (Title_without_report['Title'].str.find('soc') >= 0)| (Title_without_report['Title'].str.find('respons') >= 0)]
    print('no report num is {}, code num is {}, policy num is {}, ethic num is {}, res is {}'.format(Title_without_report.shape[0], Title_with_code.shape[0], Title_with_policy.shape[0], Title_with_ethics.shape[0], t.shape[0]))
    Title_without_report.to_csv('t.csv',index=False)

# test()
# format_source()
# classify()


def join_source_schema_by_sourceid(source, schema):
    sourceid = schema[schema['Path'].str.find('SourceId')>=0]
    ValueScore = schema[schema['Path'].str.find('ValueScore')>=0]
    ValueScore = ValueScore[['ObjectId', 'Path', 'Value']]
    sourceid = sourceid[['ObjectId', 'Path', 'Value']]
    sourceid['extra'] = sourceid['Path'].apply(lambda st: st[0:st.find('/Sources')])
    # t = t[t['Path'].str.find('SourceId') >= 0]
    ValueScore['extra'] = ValueScore['Path'].apply(lambda st: st[0:st.find('/ValueScore')])
    join = pd.merge(ValueScore,sourceid,how="left", on=['ObjectId','extra'])
    join = join.drop(labels=['extra','Path_y'], axis=1)
    join = join.rename(columns={'Value_y':'SourceId','Path_x':'Path','Value_x':'ValueScore'})
    join['SourceId'] = join['SourceId'].fillna('null')
    join = join[join['SourceId'] != 'null']    

    join = pd.merge(join,source,how="left", left_on='SourceId',right_on='ObjectId')

    join = join.drop(labels=['ObjectId_y'], axis=1)
    return join

def main(source_path, schema_path):
    # read source
    source = pd.read_csv(source_path,error_bad_lines=False)
    SourceAbstract = source[source['Path'] == 'Sources/SourceAbstract'][['ObjectId','Value']]
    SourcePublisher= source[source['Path'] == 'Sources/SourcePublisher'][['ObjectId','Value']]
    SourceUrl= source[source['Path'] == 'Sources/SourceUrl'][['ObjectId','Value']]
    SourceTitle = source[source['Path'] == 'Sources/SourceTitle'][['ObjectId','Value']]
    SourceAbstract = pd.merge(SourceAbstract,SourcePublisher,how="left", on="ObjectId")
    SourceAbstract = pd.merge(SourceAbstract,SourceUrl,how="left", on="ObjectId")
    SourceAbstract = pd.merge(SourceAbstract,SourceTitle,how="left", on="ObjectId")
    SourceAbstract.columns = ['ObjectId', 'Abstract', 'Publisher', 'Url', 'Title']
    SourceAbstract['ObjectId'] = SourceAbstract['ObjectId'].astype(str)
    print('handle source done')
    # read scheme
    scheme = pd.read_csv(schema_path,error_bad_lines=False, keep_default_na=False)
    # scheme = pd.read_csv('schema_path',error_bad_lines=False,keep_default_na=False)
    scheme = scheme[scheme['Value'] != 'null']
    print('handle schema done')

    # join source and schema
    res = join_source_schema_by_sourceid(SourceAbstract, scheme)
    print('join source and schema done')

    
    res.to_csv('res', index=False)
    print('dump res done')



    # classify
    Title = res[['Publisher', 'Title']].copy()
    Title['Title'] = Title['Title'].str.lower()
    Title['Publisher'] = Title['Publisher'].str.lower()
    # Title.drop_duplicates(inplace=True)
    # filter(Title, ['ltd', 'cor', 'company', 'inc', 'plc']).to_csv('t.csv',index=False)
    Title_with_report = Title[Title['Title'].str.find('report') >= 0]
    Title_with_annual_report = Title_with_report[Title_with_report['Title'].str.find('annual') >= 0]

    print('total num is {0}'.format(Title.shape[0]))
    print('report num is {}, annual have {}'.format(Title_with_report.shape[0], Title_with_annual_report.shape[0]))
    
    # Title_with_report.to_csv('t.csv',index=False)
    
    Title_without_report = Title[Title['Title'].str.find('report') < 0]
    Title_with_code = Title_without_report[Title_without_report['Title'].str.find('code') >= 0]
    Title_with_policy = Title_without_report[Title_without_report['Title'].str.find('policy') >= 0]
    Title_with_ethics = Title_without_report[Title_without_report['Title'].str.find('ethic') >= 0]
    # t = Title_without_report[(Title_without_report['Title'].str.find('sustai') >= 0) | (Title_without_report['Title'].str.find('gov') >= 0) | (Title_without_report['Title'].str.find('env') >= 0) | (Title_without_report['Title'].str.find('soc') >= 0)| (Title_without_report['Title'].str.find('respons') >= 0)]
    print('no report num is {}, code num is {}, policy num is {}, ethic num is {}'.format(Title_without_report.shape[0], Title_with_code.shape[0], Title_with_policy.shape[0], Title_with_ethics.shape[0]))

def add_organization_info(source_path):
    source = pd.read_csv(source_path)
    source = source.rename(columns={'ObjectId_x': 'ObjectId'})
    source['ObjectId'] = source['ObjectId'].apply(lambda t: t[0:t.find(';')])
    # print(source)


    organization_path = 'RFT-ESG-Symbology-Organization-Init-2022-09-22T10_31_35.738Z.jsonl'
    organization_info = pd.read_json(organization_path, lines=True)
    organization_info = organization_info.fillna('')
    organization_info['OrganizationName'] = organization_info['Names'].apply(lambda st: st['Name']['OrganizationName'][0]['OrganizationNormalizedName'])
    organization_info['SicSector'] = organization_info['ClassificationSicSectorSchema'].apply(lambda st: st['Industry']['IndustryName'][0]['IndustryDescription'] if st and len(st['Industry']['IndustryName']) > 0 else '')
    organization_info['NaicsPrimary'] = organization_info['ClassificationNaicsPrimarySchema'].apply(lambda st: st['Industry']['IndustryName'][0]['IndustryDescription'] if st and len(st['Industry']['IndustryName']) > 0 else '')
    organization_info['TrbcPrimary'] = organization_info['ClassificationTrbcPrimarySchema'].apply(lambda st: st['Industry']['IndustryName'][0]['IndustryDescription'] if st and len(st['Industry']['IndustryName']) > 0 else '')
    # organization_info[['ObjectId', 'OrganizationName', 'SicSector', 'NaicsPrimary', 'TrbcPrimary']].to_csv('aoligei')
    organization_info['ObjectId'] = organization_info['ObjectId'].astype(str)


    res = pd.merge(source, organization_info[['ObjectId', 'OrganizationName', 'SicSector', 'NaicsPrimary', 'TrbcPrimary']], how="left", on="ObjectId")
    res.to_csv('res')

    # print(organization_info[['OrganizationName']].info())

# add_organization_info('res.csv')
source = pd.read_csv('/usr/yubomai/esg/sources/all.csv', error_bad_lines=False)
source = source.drop_duplicates(subset=['ObjectId'], keep='last')

print(source)