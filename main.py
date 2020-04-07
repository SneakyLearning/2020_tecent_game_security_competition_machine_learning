import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def get_train_data():
    black = pd.read_table('label_black/20200301.txt',header=None,names=['uin','label'],index_col=None,sep='|')
    black['label'] = 1
    white = pd.read_table('label_white/20200301.txt',header=None,names=['uin','label'],index_col=None,sep='|')
    white['label'] = 0
    label = pd.concat([black,white])
    login = pd.read_table('role_login/20200301.txt',header=None,names=['dteventtime', 'platid', 'areaid', 'worldid', 'uin', 'roleid','rolename', 'job','rolelever', 'power', 'friendsnum', 'network', 'clientip','deviceid'],index_col=None,sep='|')
    login = login.drop(['rolename', 'deviceid'], axis=1)
    login = pd.merge(label, login, on='uin', how='left')
    logout = pd.read_table('role_logout/20200301.txt', header=None,names=['dteventtime', 'platid', 'areaid', 'worldid','uin', 'roleid', 'rolename', 'job','rolelever', 'power', 'friendsnum', 'network','clientip', 'deviceid','onlinetime'], index_col=None,sep='|')
    logout = pd.DataFrame(logout,columns=['uin','onlinetime'])
    alltime = logout.groupby('uin').sum()
    chat = pd.read_table('uin_chat/20200301.txt', header=None,
                         names=['uin', 'chat_cnt'], index_col=None, sep='|')
    train = pd.merge(login, alltime, on='uin', how='left')
    train = train.drop(['dteventtime', 'clientip'], axis=1)
    network_mapping = {network: idx for idx, network in
                       enumerate(set(train['network']))}
    train['network'] = train['network'].map(network_mapping)
    train = train.fillna(train['onlinetime'].mean())
    train = pd.merge(train, chat, on='uin', how='left')
    train = train.fillna(train['chat_cnt'].mean())
    return train

def get_result_data():
    login = pd.read_table('role_login/20200301.txt', header=None,
                          names=['dteventtime', 'platid', 'areaid', 'worldid',
                                 'uin', 'roleid', 'rolename', 'job',
                                 'rolelever', 'power', 'friendsnum', 'network',
                                 'clientip', 'deviceid'], index_col=None,
                          sep='|')
    login = login.drop(['rolename', 'deviceid'], axis=1)
    logout = pd.read_table('role_logout/20200301.txt', header=None,
                           names=['dteventtime', 'platid', 'areaid', 'worldid',
                                  'uin', 'roleid', 'rolename', 'job',
                                  'rolelever', 'power', 'friendsnum', 'network',
                                  'clientip', 'deviceid', 'onlinetime'],
                           index_col=None, sep='|')
    logout = pd.DataFrame(logout, columns=['uin', 'onlinetime'])
    alltime = logout.groupby('uin').sum()
    chat = pd.read_table('uin_chat/20200301.txt', header=None,
                         names=['uin', 'chat_cnt'], index_col=None, sep='|')
    result = pd.merge(login, alltime, on='uin', how='left')
    result = result.drop(['dteventtime', 'clientip'], axis=1)
    network_mapping = {network: idx for idx, network in
                       enumerate(set(result['network']))}
    result['network'] = result['network'].map(network_mapping)
    result = result.fillna(result['onlinetime'].mean())
    result = pd.merge(result, chat, on='uin', how='left')
    result = result.fillna(result['chat_cnt'].mean())
    return result

def train_model(data):
    y=data['label']
    x=data.drop(['label','uin'],axis=1)
    x_train,x_test,y_train,y_test = train_test_split(x,y)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    print(estimator.score(x_test,y_test))
    print(classification_report(y_test,estimator.predict(x_test)))
    return transfer,estimator

def predict(transfer,estimator,result):
    uin_result = result['uin']
    result = result.drop(['uin'], axis=1)
    result = transfer.transform(result)
    predict = estimator.predict(result)
    predict = pd.DataFrame(predict)
    laugh = pd.concat([uin_result,predict],axis=1)
    laugh.to_csv('studio.txt',sep='|',index=False,header=False)

if __name__ == "__main__":
    train = get_train_data()
    result = get_result_data()
    transfer,estimator = train_model(train)
    predict(transfer,estimator,result)