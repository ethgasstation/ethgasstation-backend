#analysis:  Run poission regression models

import json
import math
import os
import re
import sys
import subprocess
import urllib

import mysql.connector
import numpy as np 
import pandas as pd

import statsmodels.api as sm

from sqlalchemy import create_engine 
from patsy import dmatrices


from egs.settings import get_settings_filepath, load_settings, get_setting

settings_file = get_settings_filepath(os.path.dirname(os.path.realpath(__file__)))
load_settings(settings_file)

# TODO: choose one mysql or the other
cnx = mysql.connector.connect(
    user=get_setting('mysql', 'username'),
    password=get_setting('mysql', 'password'),
    host=get_setting('mysql', 'hostname'),
    port=get_setting('mysql', 'port'),
    database=get_setting('mysql', 'database'))
cursor = cnx.cursor()
engine = create_engine(
    "mysql+mysqlconnector://%s:%s@%s:%s/%s" % (
        get_setting('mysql', 'username'),
        get_setting('mysql', 'password'),
        get_setting('mysql', 'hostname'),
        get_setting('mysql', 'port'),
        get_setting('mysql', 'database')
    ), echo=False)

query = ("SELECT * FROM minedtx2")
cursor.execute(query)
head = cursor.column_names
predictData = pd.DataFrame(cursor.fetchall())
predictData.columns = head
cursor.close()


#predictData = predictData.combine_first(postedData)
predictData['confirmTime'] = predictData['block_mined']-predictData['block_posted']
print('num with confirm times')
print (predictData['confirmTime'].count())
print ('neg confirm time')
print (len(predictData.loc[predictData['confirmTime']<0]))
print ('zero confirm time')
print (len(predictData.loc[predictData['confirmTime']==0]))
print('pre-chained ' + str(len(predictData)))
predictData.loc[predictData['chained']==1, 'confirmTime']=np.nan
predictData = predictData.dropna(subset=['confirmTime'])
print('post-chained ' + str(len(predictData)))
predictData = predictData.loc[predictData['confirmTime']>0]
predictData = predictData.loc[predictData['tx_atabove']>0]
print ('cleaned transactions: ')
print (len(predictData))

print('gas offered data')
max_gasoffered = predictData['gas_offered'].max()
print('max :'+str(predictData['gas_offered'].max()))
print('delat at max')
print(predictData.loc[predictData['gas_offered'] == max_gasoffered, 'confirmTime'].values[0])
quantiles= predictData['gas_offered'].quantile([.5, .75, .95, .99])
print(quantiles)

#dep['gasCat1'] = (txData2['gasused'] == 21000).astype(int)
predictData['gasCat1'] = ((predictData['gas_offered']<=quantiles[.5])).astype(int)
predictData['gasCat2'] = ((predictData['gas_offered']>quantiles[.5]) & (predictData['gas_offered']<=quantiles[.75])).astype(int)
predictData['gasCat3'] = ((predictData['gas_offered']>quantiles[.75]) & (predictData['gas_offered']<=quantiles[.95])).astype(int)
predictData['gasCat4'] = ((predictData['gas_offered']>quantiles[.95]) & (predictData['gas_offered']<quantiles[.99])).astype(int)
predictData['gasCat5'] = (predictData['gas_offered']>=quantiles[.99]).astype(int)



predictData['hpa2'] = predictData['hashpower_accepting']*predictData['hashpower_accepting']



y, X = dmatrices('confirmTime ~ hashpower_accepting + highgas2 + tx_atabove', data = predictData, return_type = 'dataframe')

print(y[:5])
print(X[:5])

model = sm.GLM(y, X, family=sm.families.Poisson())
results = model.fit()
print (results.summary())


y['predict'] = results.predict()
y['round_gp_10gwei'] = predictData['round_gp_10gwei']
y['hashpower_accepting'] = predictData['hashpower_accepting']
y['tx_atabove'] = predictData['tx_atabove']
y['tx_unchained'] = predictData['tx_unchained']
y['highgas2'] = predictData['highgas2']


print(y)
