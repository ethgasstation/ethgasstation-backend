import pandas as pd
import numpy as np
from web3 import Web3, HTTPProvider
web3 = Web3(HTTPProvider('http://localhost:8545'))
from egs import *
import modelparams

def get_txhases_from_txpool(block):
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


def check_recent(gasprice, submitted_recent):
    """gets the %of transactions unmined submitted in recent blocks"""

    submitted_recent.loc[(submitted_recent['still_here'] >= 1) & (submitted_recent['still_here'] <= 2) & (submitted_recent['total'] < 4), 'pct_unmined'] = np.nan
    maxval = submitted_recent.loc[submitted_recent.index > gasprice, 'pct_unmined'].max()    
    if gasprice in submitted_recent.index:
        stillh = submitted_recent.get_value(gasprice, 'still_here')
        if stillh > 2:
            rval =  submitted_recent.get_value(gasprice, 'pct_unmined')
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
        '''
        if not np.isnan(row['s5mago']):
            sum1 = (modelparams.INT2 + (row['hashpower_accepting'] * modelparams.HPA2) + (row['tx_atabove'] * modelparams.TXATAB2) + (row['s5mago'] * modelparams.S5MAGO) + (row['highgas2'] *  modelparams.HIGHGAS2))

        else:
        '''
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

def analyze_last100blocks(block, alltx):
    recent_blocks = alltx.loc[alltx['block_mined'] > (block-100), ['block_mined', 'round_gp_10gwei']].copy()
    hpower = recent_blocks.groupby('round_gp_10gwei').count()
    hpower = hpower.rename(columns={'block_mined':'count'})
    totaltx  = len(recent_blocks)
    hpower['cum_tx'] = hpower['count'].cumsum()
    hpower['hashp_pct'] = hpower['cum_tx']/totaltx*100
    return(hpower)

def analyze_last5blocks(block, alltx):
    recent_blocks= alltx.loc[(alltx['block_mined'] >= (block-5)) & (alltx['block_mined'] <= (block))]
    gp_mined_10th = recent_blocks['gas_price'].quantile(.1)
    print (gp_mined_10th/1e8)
    return gp_mined_10th/1e8

def analyze_txpool(block, txpool, alltx, hashpower, avg_timemined, gaslimit,submitted_5mago, submitted_hourago, hpower):
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
    predictTable['hashpower_accepting2'] = predictTable['gasprice'].apply(get_hpa, args=(hpower,))
    
    predictTable['tx_atabove'] = predictTable['gasprice'].apply(get_tx_atabove, args=(txpool_by_gp,))
    if not submitted_5mago.empty:
        predictTable['s5mago'] = predictTable['gasprice'].apply(check_recent, args= (submitted_5mago,))
        s5mago_lookup = predictTable.set_index('gasprice')['s5mago'].to_dict()
    else:
        s5mago_lookup = None
    if not submitted_hourago.empty:
        predictTable['s1hago'] = predictTable['gasprice'].apply(check_recent, args= (submitted_hourago,))
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
    gp_lookup2 = predictTable.set_index('gasprice')['hashpower_accepting2'].to_dict()
    txatabove_lookup = predictTable.set_index('gasprice')['tx_atabove'].to_dict()

    #finally, analyze txpool transactions

    print('txpool block length ' + str(len(txpool_block)))
    txpool_block['pct_limit'] = txpool_block['gas_offered'].apply(lambda x: x / gaslimit)
    txpool_block['high_gas_offered'] = (txpool_block['pct_limit'] > modelparams.HIGHGAS1).astype(int)
    txpool_block['highgas2'] = (txpool_block['pct_limit'] > modelparams.HIGHGAS2).astype(int)
    txpool_block['hashpower_accepting'] = txpool_block['round_gp_10gwei'].apply(lambda x: gp_lookup[x] if x in gp_lookup else 100)
    txpool_block['hashpower_accepting2'] = txpool_block['round_gp_10gwei'].apply(lambda x: gp_lookup2[x] if x in gp_lookup2 else 100)
    txpool_block['hgXhpa'] = txpool_block['highgas2']*txpool_block['hashpower_accepting']
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

def make_recent_blockdf(recentdf, current_txpool):
    '''creates df for monitoring recentlymined tx'''

    def roundresult(row):
        x = row[0] / row[1] * 100
        return np.round(x)

    recentdf['still_here'] = recentdf.index.isin(current_txpool.index).astype(int)
    recentdf = recentdf[['gas_price', 'round_gp_10gwei', 'still_here']].groupby('round_gp_10gwei').agg({'gas_price':'count', 'still_here':'sum'})
    recentdf.rename(columns={'gas_price':'total'}, inplace=True)
    recentdf['pct_unmined'] = recentdf['still_here']/recentdf['total']
    recentdf['pct_unmined'] = recentdf[['still_here', 'total']].apply(roundresult, axis=1)
    return recentdf