import pandas as pd
def join_source_schema_by_sourceid(source, schema):
    sourceid = schema[schema['Path'].str.find('SourceId')>=0]
    ValueScore = schema[schema['Path'].str.endswith('Value')]
    ValueScore = ValueScore[['ObjectId', 'Path', 'Value']]
    sourceid = sourceid[['ObjectId', 'Path', 'Value']]
    sourceid['extra'] = sourceid['Path'].apply(lambda st: st[0:st.find('/Sources')])
    # t = t[t['Path'].str.find('SourceId') >= 0]
    ValueScore['extra'] = ValueScore['Path'].apply(lambda st: st[0:st.find('/Value')])
    join = pd.merge(ValueScore,sourceid,how="left", on=['ObjectId','extra'])
    join = join.drop(labels=['extra','Path_y'], axis=1)
    join = join.rename(columns={'Value_y':'SourceId','Path_x':'Path','Value_x':'Value'})
    join['SourceId'] = join['SourceId'].fillna('null')
    join = join[join['SourceId'] != 'null']    
    join = join[join['Value'] != 'true']
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



    # # classify
    # Title = res[['Publisher', 'Title']].copy()
    # Title['Title'] = Title['Title'].str.lower()
    # Title['Publisher'] = Title['Publisher'].str.lower()
    # # Title.drop_duplicates(inplace=True)
    # # filter(Title, ['ltd', 'cor', 'company', 'inc', 'plc']).to_csv('t.csv',index=False)
    # Title_with_report = Title[Title['Title'].str.find('report') >= 0]
    # Title_with_annual_report = Title_with_report[Title_with_report['Title'].str.find('annual') >= 0]

    # print('total num is {0}'.format(Title.shape[0]))
    # print('report num is {}, annual have {}'.format(Title_with_report.shape[0], Title_with_annual_report.shape[0]))
    
    # # Title_with_report.to_csv('t.csv',index=False)
    
    # Title_without_report = Title[Title['Title'].str.find('report') < 0]
    # Title_with_code = Title_without_report[Title_without_report['Title'].str.find('code') >= 0]
    # Title_with_policy = Title_without_report[Title_without_report['Title'].str.find('policy') >= 0]
    # Title_with_ethics = Title_without_report[Title_without_report['Title'].str.find('ethic') >= 0]
    # # t = Title_without_report[(Title_without_report['Title'].str.find('sustai') >= 0) | (Title_without_report['Title'].str.find('gov') >= 0) | (Title_without_report['Title'].str.find('env') >= 0) | (Title_without_report['Title'].str.find('soc') >= 0)| (Title_without_report['Title'].str.find('respons') >= 0)]
    # print('no report num is {}, code num is {}, policy num is {}, ethic num is {}'.format(Title_without_report.shape[0], Title_with_code.shape[0], Title_with_policy.shape[0], Title_with_ethics.shape[0]))



main('/usr/yubomai/esg/sources/all.csv', '/usr/yubomai/esg/schemeB/env')