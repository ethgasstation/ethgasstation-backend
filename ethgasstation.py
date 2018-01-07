#!/usr/bin/env python3
"""
ethgasstation.py

EthGasStation adaptive oracle.
For more information, see README.md.
"""

import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DECIMAL, BigInteger, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from web3 import Web3, HTTPProvider

# internal libraries
from egs.egs_ref import *
from egs.settings import load_settings, get_setting, get_settings_filepath
from egs.jsonexporter import JSONExporter

import egs.modelparams as modelparams

# load settings from configuration
settings_file = get_settings_filepath(os.path.dirname(os.path.realpath(__file__)))
load_settings(settings_file)

json_output_dir = os.path.abspath(get_setting('json', 'output_directory'))
if not os.path.isdir(json_output_dir):
    # XXX should be on stderr
    print("WARN: Could not find output directory %s" % json_output_dir)
    print("WARN: Making directory tree.")
    # attempt to make dirs
    os.makedirs(json_output_dir, exist_ok=True)

# configure necessary services
web3 = Web3(
    HTTPProvider(
        "%s://%s:%s" % (
            get_setting('geth', 'protocol'),
            get_setting('geth', 'hostname'),
            get_setting('geth', 'port'))))

connstr = "mysql+mysqlconnector://%s:%s@%s:%s/%s" % (
        get_setting('mysql', 'username'),
        get_setting('mysql', 'password'),
        get_setting('mysql', 'hostname'),
        get_setting('mysql', 'port'),
        get_setting('mysql', 'database')
    )
engine = create_engine(connstr, echo=False)

# create tables, sql session
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()



def init_dfs():
    """load data from mysql"""
    blockdata = pd.read_sql('SELECT * from blockdata2 order by id desc limit 2000', con=engine)
    blockdata = blockdata.drop('id', axis=1)
    postedtx = pd.read_sql('SELECT * from postedtx2 order by id desc limit 100000', con=engine)
    minedtx = pd.read_sql('SELECT * from minedtx2 order by id desc limit 100000', con=engine)
    minedtx.set_index('index', drop=True, inplace=True)
    alltx = pd.read_sql('SELECT * from minedtx2 order by id desc limit 100000', con=engine)
    alltx.set_index('index', drop=True, inplace=True)
    alltx = postedtx[['index', 'expectedTime', 'expectedWait', 'mined_probability', 'highgas2', 'from_address', 'gas_offered', 'gas_price', 'hashpower_accepting', 'num_from', 'num_to', 'ico', 'dump', 'high_gas_offered', 'pct_limit', 'round_gp_10gwei', 'time_posted', 'block_posted', 'to_address', 'tx_atabove', 'wait_blocks', 'chained', 'nonce']].join(minedtx[['block_mined', 'miner', 'time_mined', 'removed_block']], on='index', how='left')
    alltx.set_index('index', drop=True, inplace=True)
    return(blockdata, alltx)

def prune_data(blockdata, alltx, txpool, block):
    """keep dataframes and databases from getting too big"""
    stmt = text("DELETE FROM postedtx2 WHERE block_posted <= :block")
    stmt2 = text("DELETE FROM minedtx2 WHERE block_mined <= :block")
    deleteBlock_sql = block - 5500
    deleteBlock_mined = block - 1700
    deleteBlock_posted = block - 5500
    engine.execute(stmt, block=deleteBlock_sql)
    engine.execute(stmt2, block=deleteBlock_sql)
    alltx = alltx.loc[(alltx['block_posted'] > deleteBlock_posted) | (alltx['block_mined'] > deleteBlock_mined)]
    blockdata = blockdata.loc[blockdata['block_number'] > deleteBlock_posted]
    txpool = txpool.loc[txpool['block'] > (block-10)]
    return (blockdata, alltx, txpool)

def write_to_sql(alltx, analyzed_block, block_sumdf, mined_blockdf, block):
    """write data to mysql for analysis"""
    post = alltx[alltx.index.isin(mined_blockdf.index)]
    post.to_sql(con=engine, name='minedtx2', if_exists='append', index=True)
    print ('num mined = ' + str(len(post)))
    post2 = alltx.loc[alltx['block_posted'] == (block-1)]
    post2.to_sql(con=engine, name='postedtx2', if_exists='append', index=True)
    print ('num posted = ' + str(len(post2)))
    analyzed_block.to_sql(con=engine, name='txpool_current', index=False, if_exists='replace')
    block_sumdf.to_sql(con=engine, name='blockdata2', if_exists='append', index=False)

def write_to_json(gprecs, txpool_by_gp, prediction_table, analyzed_block,submitted_hourago=None):
    """write json data"""
    exporter = JSONExporter()

    try:
        txpool_by_gp = txpool_by_gp.rename(columns={'gas_price':'count'})
        txpool_by_gp['gasprice'] = txpool_by_gp['round_gp_10gwei']/10

        prediction_table['gasprice'] = prediction_table['gasprice']/10
        analyzed_block_show  = analyzed_block.loc[analyzed_block['chained']==0].copy()
        analyzed_block_show['gasprice'] = analyzed_block_show['round_gp_10gwei']/10
        analyzed_block_show = analyzed_block_show[['index', 'block_posted', 'gas_offered', 'gasprice', 'hashpower_accepting', 'tx_atabove', 'mined_probability', 'expectedWait', 'wait_blocks']].sort_values('wait_blocks', ascending=False)
        analyzed_blockout = analyzed_block_show.to_json(orient='records')
        prediction_tableout = prediction_table.to_json(orient='records')
        txpool_by_gpout = txpool_by_gp.to_json(orient='records')

        exporter.write_json('ethgasAPI', gprecs)
        exporter.write_json('memPool', txpool_by_gpout)
        exporter.write_json('predictTable', prediction_tableout)
        exporter.write_json('txpoolblock', analyzed_blockout)

        if not submitted_hourago.empty:
            submitted_hourago = submitted_hourago.to_json(orient='records')
            exporter.write_json('hourago', submitted_hourago)

    except Exception as e:
        print(e)

def get_txhashes_from_txpool(block):
    """gets list of all txhash in txpool at block and returns dataframe"""
    hashlist = []
    txpoolcontent = web3.txpool.content
    txpoolpending = txpoolcontent['pending']
    for tx_sequence in txpoolpending.values():
        for tx_obj in tx_sequence.values():
            hashlist.append(tx_obj['hash'])
    txpool_current = pd.DataFrame(index = hashlist)
    txpool_current['block'] = block
    return txpool_current

def process_block_transactions(block):
    """get tx data from block"""
    block_df = pd.DataFrame()
    block_obj = web3.eth.getBlock(block, True)
    miner = block_obj.miner
    for transaction in block_obj.transactions:
        clean_tx = CleanTx(transaction, None, None, miner)
        block_df = block_df.append(clean_tx.to_dataframe(), ignore_index = False)
    block_df['time_mined'] = block_obj.timestamp
    return(block_df, block_obj)

def process_block_data(block_df, block_obj):
    """process block to dataframe"""
    if len(block_obj.transactions) > 0:
        block_df['weighted_fee'] = block_df['round_gp_10gwei']* block_df['gas_offered']
        block_mingasprice = block_df['round_gp_10gwei'].min()
        block_weightedfee = block_df['weighted_fee'].sum() / block_df['gas_offered'].sum()
    else:
        block_mingasprice = np.nan
        block_weightedfee = np.nan
    block_numtx = len(block_obj.transactions)
    timemined = block_df['time_mined'].min()
    clean_block = CleanBlock(block_obj, 1, 0, timemined, block_mingasprice, block_numtx, block_weightedfee)
    return(clean_block.to_dataframe())

def get_hpa(gasprice, hashpower):
    """gets the hash power accpeting the gas price over last 200 blocks"""
    hpa = hashpower.loc[gasprice >= hashpower.index, 'hashp_pct']
    if gasprice > hashpower.index.max():
        hpa = 100
    elif gasprice < hashpower.index.min():
        hpa = 0
    else:
        hpa = hpa.max()
    return int(hpa)

def get_tx_atabove(gasprice, txpool_by_gp):
    """gets the number of transactions in the txpool at or above the given gasprice"""
    txAtAb = txpool_by_gp.loc[txpool_by_gp.index >= gasprice, 'gas_price']
    if gasprice > txpool_by_gp.index.max():
        txAtAb = 0
    else:
        txAtAb = txAtAb.sum()
    return txAtAb
def check_10th(gasprice, gp_mined_10th):
    x = np.round((gasprice - gp_mined_10th))
    x = x / 10
    return x

def check_5mago(gasprice, submitted_5mago):

    submitted_5mago.loc[(submitted_5mago['still_here'] >= 1) & (submitted_5mago['still_here'] <= 2) & (submitted_5mago['total'] < 4), 'pct_unmined'] = np.nan

    maxval = submitted_5mago.loc[submitted_5mago.index > gasprice, 'pct_unmined'].max()

    if gasprice in submitted_5mago.index:
        stillh = submitted_5mago.get_value(gasprice, 'still_here')
        if stillh > 2:
            rval =  submitted_5mago.get_value(gasprice, 'pct_unmined')
        else:
            rval = maxval
    else:
        rval = maxval

    if gasprice >= 1000:
        rval = 0

    if (rval > maxval) or (gasprice >= 1000) :
        return rval
    return maxval

def check_hourago(gasprice, submitted_hourago):

    submitted_hourago.loc[(submitted_hourago['still_here'] >= 1) & (submitted_hourago['still_here'] <= 2) & (submitted_hourago['total'] <= 5), 'pct_unmined'] = np.nan

    maxval = submitted_hourago.loc[submitted_hourago.index > gasprice, 'pct_unmined'].max()

    if gasprice in submitted_hourago.index:
        stillh = submitted_hourago.get_value(gasprice, 'still_here')
        if stillh > 2:
            rval =  submitted_hourago.get_value(gasprice, 'pct_unmined')
        else:
            rval = maxval
    else:
        rval = maxval

    if gasprice >= 1000:
        rval = 0

    if (rval > maxval) or (gasprice >= 1000) :
        return rval
    return maxval



def predict(row):
    if row['chained'] == 1:
        return np.nan

    #set in modelparams
    try:
        sum1 = (modelparams.INTERCEPT + (row['hashpower_accepting'] * modelparams.HPA_COEF) + (row['tx_atabove'] * modelparams.TXATABOVE_COEF) + (row['hgXhpa'] * modelparams.INTERACT_COEF) + (row['highgas2'] *  modelparams.HIGHGAS_COEF))
        prediction = np.exp(sum1)
        if prediction < 2:
            prediction = 2
        if row['gas_offered'] > 2000000:
            prediction = prediction + 100
        return np.round(prediction, decimals=2)
    except Exception as e:
        print(e)
        return np.nan

def check_nonce(row, txpool_block_nonce):
    if row['num_from']>1:
        if row['nonce'] > txpool_block_nonce.loc[row['from_address'],'nonce']:
            return 1
        if row['nonce'] == txpool_block_nonce.loc[row['from_address'], 'nonce']:
            return 0
    else:
        return 0

def analyze_last200blocks(block, blockdata):
    recent_blocks = blockdata.loc[blockdata['block_number'] > (block-200), ['mingasprice', 'block_number', 'gaslimit', 'time_mined', 'speed']]
    gaslimit = recent_blocks['gaslimit'].mean()
    last10 = recent_blocks.sort_values('block_number', ascending=False).head(n=10)
    speed = last10['speed'].mean()
    #create hashpower accepting dataframe based on mingasprice accepted in block
    hashpower = recent_blocks.groupby('mingasprice').count()
    hashpower = hashpower.rename(columns={'block_number': 'count'})
    hashpower['cum_blocks'] = hashpower['count'].cumsum()
    totalblocks = hashpower['count'].sum()
    hashpower['hashp_pct'] = hashpower['cum_blocks']/totalblocks*100
    #get avg blockinterval time
    blockinterval = recent_blocks.sort_values('block_number').diff()
    blockinterval.loc[blockinterval['block_number'] > 1, 'time_mined'] = np.nan
    blockinterval.loc[blockinterval['time_mined']< 0, 'time_mined'] = np.nan
    avg_timemined = blockinterval['time_mined'].mean()
    if np.isnan(avg_timemined):
        avg_timemined = 30
    return(hashpower, avg_timemined, gaslimit, speed)

def analyze_last5blocks(block, alltx):
    recent_blocks= alltx.loc[(alltx['block_mined'] >= (block-5)) & (alltx['block_mined'] <= (block))]
    gp_mined_10th = recent_blocks['gas_price'].quantile(.1)
    print (gp_mined_10th/1e8)
    return gp_mined_10th/1e8


def analyze_txpool(block, txpool, alltx, hashpower, avg_timemined, gaslimit, gp_mined_10th, submitted_5mago, submitted_hourago):
    """gets txhash from all transactions in txpool at block and merges the data from alltx"""
    #get txpool hashes at block
    txpool_block = txpool.loc[txpool['block']==block]
    if (len(txpool_block)==0):
        return(pd.DataFrame(), None, None)
    txpool_block = txpool_block.drop(['block'], axis=1)
    #merge transaction data for txpool transactions
    #txpool_block only has transactions received by filter
    txpool_block = txpool_block.join(alltx, how='inner')

    #txpool_block = txpool_block[~txpool_block.index.duplicated(keep = 'first')]
    assert txpool_block.index.duplicated(keep='first').sum() == 0

    txpool_block['num_from'] = txpool_block.groupby('from_address')['block_posted'].transform('count')
    txpool_block['num_to'] = txpool_block.groupby('to_address')['block_posted'].transform('count')
    txpool_block['ico'] = (txpool_block['num_to'] > 90).astype(int)
    txpool_block['dump'] = (txpool_block['num_from'] > 5).astype(int)

    #new dfs grouped by gasprice and nonce
    txpool_by_gp = txpool_block[['gas_price', 'round_gp_10gwei']].groupby('round_gp_10gwei').agg({'gas_price':'count'})
    txpool_block_nonce = txpool_block[['from_address', 'nonce']].groupby('from_address').agg({'nonce':'min'})

    #when multiple tx from same from address, finds tx with lowest nonce (unchained) - others are 'chained'
    txpool_block['chained'] = txpool_block.apply(check_nonce, args=(txpool_block_nonce,), axis=1)

    #predictiontable
    predictTable = pd.DataFrame({'gasprice' :  range(10, 1010, 10)})
    ptable2 = pd.DataFrame({'gasprice' : range(0, 10, 1)})
    predictTable = predictTable.append(ptable2).reset_index(drop=True)
    predictTable = predictTable.sort_values('gasprice').reset_index(drop=True)
    predictTable['hashpower_accepting'] = predictTable['gasprice'].apply(get_hpa, args=(hashpower,))
    predictTable['tx_atabove'] = predictTable['gasprice'].apply(get_tx_atabove, args=(txpool_by_gp,))
    predictTable['gp10th'] = predictTable['gasprice'].apply(check_10th, args=(gp_mined_10th,))
    if not submitted_5mago.empty:
        predictTable['s5mago'] = predictTable['gasprice'].apply(check_5mago, args= (submitted_5mago,))
        s5mago_lookup = predictTable.set_index('gasprice')['s5mago'].to_dict()
    else:
        s5mago_lookup = None
    if not submitted_hourago.empty:
        predictTable['s1hago'] = predictTable['gasprice'].apply(check_hourago, args= (submitted_hourago,))
        s1hago_lookup = predictTable.set_index('gasprice')['s1hago'].to_dict()
    else:
        s1hago_lookup = None
    predictTable['ico'] = 0
    predictTable['dump'] = 0
    predictTable['gas_offered'] = 0
    predictTable['wait_blocks'] = 0
    predictTable['highgas2'] = 0
    predictTable['chained'] = 0
    predictTable['hgXhpa'] = 0
    predictTable['wait_blocks'] = 0
    predictTable['expectedWait'] = predictTable.apply(predict, axis=1)
    predictTable['expectedTime'] = predictTable['expectedWait'].apply(lambda x: np.round((x * avg_timemined / 60), decimals=2))
    gp_lookup = predictTable.set_index('gasprice')['hashpower_accepting'].to_dict()
    txatabove_lookup = predictTable.set_index('gasprice')['tx_atabove'].to_dict()

    #finally, analyze txpool transactions
    print('txpool block length ' + str(len(txpool_block)))
    txpool_block['pct_limit'] = txpool_block['gas_offered'].apply(lambda x: x / gaslimit)
    txpool_block['high_gas_offered'] = (txpool_block['pct_limit'] > modelparams.HIGHGAS1).astype(int)
    txpool_block['highgas2'] = (txpool_block['pct_limit'] > modelparams.HIGHGAS2).astype(int)
    txpool_block['hashpower_accepting'] = txpool_block['round_gp_10gwei'].apply(lambda x: gp_lookup[x] if x in gp_lookup else 100)
    txpool_block['hgXhpa'] = txpool_block['highgas2']*txpool_block['hashpower_accepting']
    txpool_block['gp10th'] = txpool_block['round_gp_10gwei'].apply(check_10th, args=(gp_mined_10th,))
    txpool_block['tx_atabove'] = txpool_block['round_gp_10gwei'].apply(lambda x: txatabove_lookup[x] if x in txatabove_lookup else 1)
    if not s5mago_lookup is None:
        txpool_block['s5mago'] = txpool_block['round_gp_10gwei'].apply(lambda x: s5mago_lookup[x] if x in s5mago_lookup else np.nan)
        txpool_block.loc[txpool_block['round_gp_10gwei'] >= 100, 's5mago'] = 0
    else:
        txpool_block['s5mago'] = np.nan
    if not s1hago_lookup is None:
        txpool_block['s1hago'] = txpool_block['round_gp_10gwei'].apply(lambda x: s1hago_lookup[x] if x in s1hago_lookup else np.nan)
        txpool_block.loc[txpool_block['round_gp_10gwei'] >= 100, 's1hago'] = 0
    else:
        txpool_block['s1hago'] = np.nan
    txpool_block['expectedWait'] = txpool_block.apply(predict, axis=1)
    txpool_block['expectedTime'] = txpool_block['expectedWait'].apply(lambda x: np.round((x * avg_timemined / 60), decimals=2))
    txpool_by_gp = txpool_block[['gas_price', 'round_gp_10gwei']].groupby('round_gp_10gwei').agg({'gas_price':'count'})
    txpool_by_gp.reset_index(inplace=True, drop=False)
    return(txpool_block, txpool_by_gp, predictTable)

def get_gasprice_recs(prediction_table, block_time, block, speed, minlow=-1, submitted_hourago=None):

    def get_safelow(minlow, submitted_hourago):
        series = prediction_table.loc[prediction_table['expectedTime'] <= 30, 'gasprice']
        safelow = series.min()
        minhash_list = prediction_table.loc[prediction_table['hashpower_accepting']>=1.5, 'gasprice']
        if (safelow < minhash_list.min()):
            safelow = minhash_list.min()
        if minlow >= 0:
             if safelow < minlow:
                safelow = minlow

        return float(safelow)

    def get_average(safelow):
        series = prediction_table.loc[prediction_table['expectedTime'] <= 5, 'gasprice']
        average = series.min()
        minhash_list = prediction_table.loc[prediction_table['hashpower_accepting']>20, 'gasprice']
        if average < minhash_list.min():
            average = minhash_list.min()
        if average < safelow:
            average = safelow
        if np.isnan(average):
            average = 500
        return float(average)

    def get_fast():
        series = prediction_table.loc[prediction_table['expectedTime'] <= 1, 'gasprice']
        fastest = series.min()
        minhash_list = prediction_table.loc[prediction_table['hashpower_accepting']>90, 'gasprice']
        if fastest < minhash_list.min():
            fastest = minhash_list.min()
        if np.isnan(fastest):
            fastest = 1000
        return float(fastest)

    def get_fastest():
        fastest = prediction_table['expectedTime'].min()
        series = prediction_table.loc[prediction_table['expectedTime'] == fastest, 'gasprice']
        fastest = series.min()
        minhash_list = prediction_table.loc[prediction_table['hashpower_accepting']>95, 'gasprice']
        if fastest < minhash_list.min():
            fastest = minhash_list.min()
        return float(fastest)

    def get_wait(gasprice):
        try:
            wait =  prediction_table.loc[prediction_table['gasprice']==gasprice, 'expectedTime'].values[0]
        except:
            wait = 0
        wait = round(wait, 1)
        return float(wait)

    gprecs = {}
    gprecs['safeLow'] = get_safelow(minlow, submitted_hourago)
    gprecs['safeLowWait'] = get_wait(gprecs['safeLow'])
    gprecs['average'] = get_average(gprecs['safeLow'])
    gprecs['avgWait'] = get_wait(gprecs['average'])
    gprecs['fast'] = get_fast()
    gprecs['fastWait'] = get_wait(gprecs['fast'])
    gprecs['fastest'] = get_fastest()
    gprecs['fastestWait'] = get_wait(gprecs['fastest'])
    gprecs['block_time'] = block_time
    gprecs['blockNum'] = block
    gprecs['speed'] = speed
    return(gprecs)

def roundresult(row):
    x = row[0] / row[1] * 100
    return np.round(x)

def master_control():
    (blockdata, alltx) = init_dfs()
    txpool = pd.DataFrame()
    snapstore = pd.DataFrame()
    print ('blocks '+ str(len(blockdata)))
    print ('txcount '+ str(len(alltx)))
    timer = Timers(web3.eth.blockNumber)
    start_time = time.time()
    tx_filter = web3.eth.filter('pending')


    def append_new_tx(clean_tx):
        nonlocal alltx
        if not clean_tx.hash in alltx.index:
            alltx = alltx.append(clean_tx.to_dataframe(), ignore_index = False)



    def update_dataframes(block):
        nonlocal alltx
        nonlocal txpool
        nonlocal blockdata
        nonlocal timer

        print('updating dataframes at block '+ str(block))
        try:
            #get minedtransactions and blockdata from previous block
            mined_block_num = block-3
            (mined_blockdf, block_obj) = process_block_transactions(mined_block_num)

            #add mined data to tx dataframe - only unique hashes seen by node
            mined_blockdf_seen = mined_blockdf[mined_blockdf.index.isin(alltx.index)]
            print('num mined in ' + str(mined_block_num)+ ' = ' + str(len(mined_blockdf)))
            print('num seen in ' + str(mined_block_num)+ ' = ' + str(len(mined_blockdf_seen)))
            alltx = alltx.combine_first(mined_blockdf)

            #process block data
            block_sumdf = process_block_data(mined_blockdf, block_obj)

            #add block data to block dataframe
            blockdata = blockdata.append(block_sumdf, ignore_index = True)

            #get list of txhashes from txpool
            current_txpool = get_txhashes_from_txpool(block)

            #add txhashes to txpool dataframe
            txpool = txpool.append(current_txpool, ignore_index = False)

            #get hashpower table, block interval time, gaslimit, speed from last 200 blocks
            (hashpower, block_time, gaslimit, speed) = analyze_last200blocks(block, blockdata)
            #get 10th percentile of mined gas prices of last 5 blocks
            gp_mined_10th = analyze_last5blocks(block, alltx)

            submitted_hourago = alltx.loc[(alltx['block_posted'] < (block-130)) & (alltx['block_posted'] > (block-260)) & (alltx['chained']==0) & (alltx['gas_offered'] < 500000)].copy()
            print(len(submitted_hourago))

            if len(submitted_hourago > 50):
                submitted_hourago['still_here'] = submitted_hourago.index.isin(current_txpool.index).astype(int)
                submitted_hourago = submitted_hourago[['gas_price', 'round_gp_10gwei', 'still_here']].groupby('round_gp_10gwei').agg({'gas_price':'count', 'still_here':'sum'})
                submitted_hourago.rename(columns={'gas_price':'total'}, inplace=True)
                submitted_hourago['pct_unmined'] = submitted_hourago['still_here']/submitted_hourago['total']
                submitted_hourago['pct_unmined'] = submitted_hourago[['still_here', 'total']].apply(roundresult, axis=1)
            else:
                submitted_hourago = pd.DataFrame()

            submitted_5mago = alltx.loc[(alltx['block_posted'] < (block-20)) & (alltx['block_posted'] > (block-70)) & (alltx['chained']==0) & (alltx['gas_offered'] < 500000)].copy()
            print(len(submitted_5mago))

            if len(submitted_5mago > 50):
                submitted_5mago['still_here'] = submitted_5mago.index.isin(current_txpool.index).astype(int)
                submitted_5mago = submitted_5mago[['gas_price', 'round_gp_10gwei', 'still_here']].groupby('round_gp_10gwei').agg({'gas_price':'count', 'still_here':'sum'})
                submitted_5mago.rename(columns={'gas_price':'total'}, inplace=True)
                submitted_5mago['pct_unmined'] = submitted_5mago[['still_here', 'total']].apply(roundresult, axis=1)
            else:
                submitted_5mago = pd.DataFrame()


            #make txpool block data
            (analyzed_block, txpool_by_gp, predictiondf) = analyze_txpool(block-1, txpool, alltx, hashpower, block_time, gaslimit, gp_mined_10th, submitted_5mago, submitted_hourago)
            if analyzed_block.empty:
                print("txpool block is empty - returning")
                return
            assert analyzed_block.index.duplicated().sum()==0
            alltx = alltx.combine_first(analyzed_block)



            #with pd.option_context('display.max_columns', None,):
                #print(analyzed_block)
            # update tx dataframe with txpool variables and time preidctions

            #get gpRecs
            gprecs = get_gasprice_recs (predictiondf, block_time, block, speed, timer.minlow, submitted_hourago)

            #every block, write gprecs, predictions, txpool by gasprice
            analyzed_block.reset_index(drop=False, inplace=True)
            if not submitted_hourago.empty:
                submitted_hourago.reset_index(drop=False, inplace=True)
                submitted_hourago = submitted_hourago.sort_values('round_gp_10gwei')
            write_to_json(gprecs, txpool_by_gp, predictiondf, analyzed_block, submitted_hourago)
            write_to_sql(alltx, analyzed_block, block_sumdf, mined_blockdf, block)

            #keep from getting too large
            (blockdata, alltx, txpool) = prune_data(blockdata, alltx, txpool, block)
            return True

        except:
            print(traceback.format_exc())


    while True:
        try:
            new_tx_list = web3.eth.getFilterChanges(tx_filter.filter_id)
        except:
            tx_filter = web3.eth.filter('pending')
            new_tx_list = web3.eth.getFilterChanges(tx_filter.filter_id)
        block = web3.eth.blockNumber
        timestamp = time.time()
        if (timer.process_block > (block - 8)):
            for new_tx in new_tx_list:
                try:
                    tx_obj = web3.eth.getTransaction(new_tx)
                    clean_tx = CleanTx(tx_obj, block, timestamp)
                    append_new_tx(clean_tx)
                except Exception as e:
                    pass
        if (timer.process_block < block):

            if block > timer.start_block+1:
                print('current block ' +str(block))
                print ('processing block ' + str(timer.process_block))
                updated = update_dataframes(timer.process_block)
                print ('finished ' + str(timer.process_block))
                timer.process_block = timer.process_block + 1


def main():
    """int main"""
    master_control()

if __name__ == "__main__":
    main()
