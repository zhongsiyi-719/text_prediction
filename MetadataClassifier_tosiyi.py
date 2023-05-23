# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:06:28 2023

@author: NAGY180
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rc("font",family='YouYuan')

# 读取数据
DataAssetCatalogue = pd.read_excel('DataAssetCatalogue (5).xlsx',
                                   sheet_name = '数据资产映射')
# 剔除单元格中换行符
DataAssetCatalogue = DataAssetCatalogue.replace('\n','').replace('\r','')

# 拼接前4列，作为我们训练和预测的target
DataAssetCatalogue['类别'] = DataAssetCatalogue['主题域名称']+'#'\
                            +DataAssetCatalogue['二级分类']+'#'\
                            +DataAssetCatalogue['三级分类']+'#'\
                            +DataAssetCatalogue['四级分类']+'#'



################# 数据探索 ####################
CatalogueCount = pd.DataFrame(DataAssetCatalogue.groupby(['类别'])['数据资产'].count().reset_index())
CatalogueCount.rename(columns={'数据资产':'数量'},inplace=True)
CatalogueCount.sort_values(by='数量',ascending=False,inplace=True)

# 只取样本数大于1的类别进行训练
modelcata = set(CatalogueCount[CatalogueCount['数量']>1]['类别'])

modeldata = DataAssetCatalogue[DataAssetCatalogue['类别'].isin(modelcata)].copy()

modeldata['fieldinfo'] = DataAssetCatalogue['源系统名称'].astype(str)+'#'\
                            +DataAssetCatalogue['源表英文名'].astype(str)+'#'\
                            +DataAssetCatalogue['源表中文名'].astype(str)+'#'\
                            +DataAssetCatalogue['源字段英文名'].astype(str)+'#'\
                            +DataAssetCatalogue['源字段中文名'].astype(str)+'#'



modeldata['fieldinfo'] = modeldata['fieldinfo'].apply(lambda x:x.replace('\n','').replace('\r',''))

modeldata['label'] = '__label__' + modeldata['类别'].astype(str)


############## 区分训练集和测试集 #################
from sklearn.model_selection import StratifiedShuffleSplit    
s = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
train_idx,test_idx = next(s.split(modeldata,modeldata['类别']))
train_data,test_data = modeldata.iloc[train_idx],modeldata.iloc[test_idx]



############### 基于FastText的文本分类 ###############
from sklearn.metrics import f1_score,precision_score,recall_score
import fasttext

# 转换为FastText需要的格式
train_df = modeldata[['fieldinfo','label']]
train_df.to_csv('train_df.csv',index=None, header=None,sep='\t')


# 建立文本分类模型
model = fasttext.train_supervised('train_df.csv', lr=1.0, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=25, loss="softmax")

val_pred = [model.predict(x)[0][0] for x in test_data['fieldinfo']]
test_data['label_pred'] = val_pred


print(f1_score(test_data['label'], val_pred, average='weighted'))
print(precision_score(test_data['label'], val_pred,average='weighted'))
print(recall_score(test_data['label'], val_pred,average='weighted'))

test_data.to_csv('test_df.csv',encoding='utf_8_sig',index=False)


############### 基于FastText的词向量训练 ###############
train_df['fieldinfo'].to_csv('train_embedding.csv',index=None, header=None,sep='\t')
modeldata.iloc[train_idx]['fieldnameinfo'].to_csv('train_embedding.csv',index=None, header=None,sep='\t')

embedding = fasttext.train_unsupervised('train_embedding.csv', epoch=5, lr=0.5,dim=50,wordNgrams=5)
words = embedding.words

############## 输出大类下词向量相似度最高的数据资产名 ###############
def PredictAssetname(text,label):
    temp = train_data[train_data['label'] == label]
    assetname = set(temp['数据资产'])
    ans = ''
    maxsim = -2
    vec1 = embedding.get_word_vector(text)
    for x in assetname:
        temp2 = temp[temp['数据资产'] == x].copy()
        for t in temp2['fieldinfo']:
            vec2 = embedding.get_word_vector(t)
            cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            if cos_sim > maxsim:
                ans,maxsim = x,cos_sim
    return ans 

test_data['数据资产_pred'] = ''
for i in test_data.index:
    test_data.loc[i,'数据资产_pred'] = PredictAssetname(test_data.loc[i,'fieldinfo'],test_data.at[i,'label_pred'])
test_data['总类别_pred'] = test_data['label_pred'] + test_data['数据资产_pred']
test_data['总类别'] = test_data['label'] + test_data['数据资产']
test_data.to_csv('test_df2.csv',encoding='utf_8_sig',index=False)

print(f1_score(test_data['总类别'], test_data['总类别_pred'], average='weighted'))  
print(precision_score(test_data['总类别'], test_data['总类别_pred'],average='weighted'))
print(recall_score(test_data['总类别'], test_data['总类别_pred'],average='weighted'))



