import pandas as pd
import numpy as np
import traceback
from egs.egs_ref import *
import egs.settings
import egs.modelparams as modelparams

egs.settings.load_settings()
web3 = egs.settings.get_web3_provider()

def get_txhashes_from_txpool(block):
    """gets list of all txhash in txpool at block and returns dataframe"""
    hashlist = []
    try:
        txpoolcontent = web3.txpool.content
        txpoolpending = txpoolcontent['pending']
        for tx_sequence in txpoolpending.values():
            for tx_obj in tx_sequence.values():
                hashlist.append(tx_obj['hash'])
        txpool_current = pd.DataFrame(index = hashlist)
        txpool_current['block'] = block
        return txpool_current
    except:
        return None


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

    #set this to avoid false positive delays
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

def get_recent_value(gasprice, submitted_recent, col):
    """gets values from recenttx df for prediction table"""
    if gasprice in submitted_recent.index:
        rval = submitted_recent.get_value(gasprice, col)
    else:
        rval = 0
    return rval


def predict(row):
    if row['chained'] == 1:
        return np.nan
    #set in modelparams

    try:
        sum1 = (modelparams.INTERCEPT + (row['hashpower_accepting'] * modelparams.HPA_COEF))
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

def make_txpool_block(block, txpool, alltx):
    """gets txhash from all transactions in txpool at block and merges the data from alltx"""
    #get txpool hashes at block
    txpool_block = txpool.loc[txpool['block']==block]
    if not txpool_block.empty:
        txpool_block = txpool_block.drop(['block'], axis=1)
        #merge transaction data for txpool transactions
        #txpool_block only has transactions received by filter
        txpool_block = txpool_block.join(alltx, how='inner')
        txpool_block = txpool_block.append(alltx.loc[alltx['block_posted']==block])
        print('txpool block length ' + str(len(txpool_block)))
    else:
        txpool_block = pd.DataFrame()
        print ('txpool block length 0')
    return txpool_block

def analyze_nonce(txpool_block, txpool_block_nonce):
    """flags tx in txpool_block that are chained to lower nonce tx"""
    txpool_block['num_from'] = txpool_block.groupby('from_address')['block_posted'].transform('count')
    #when multiple tx from same from address, finds tx with lowest nonce (unchained) - others are 'chained'
    txpool_block['chained'] = txpool_block.apply(check_nonce, args=(txpool_block_nonce,), axis=1)
    return txpool_block


def make_predcitiontable (hashpower, hpower, avg_timemined, txpool_by_gp, submitted_5mago, submitted_30mago):
    """makes prediction table for confirmations based on parameters"""
    predictTable = pd.DataFrame({'gasprice' :  range(10, 1010, 10)})
    ptable2 = pd.DataFrame({'gasprice' : range(0, 10, 1)})
    predictTable = predictTable.append(ptable2).reset_index(drop=True)
    predictTable = predictTable.sort_values('gasprice').reset_index(drop=True)
    predictTable['hashpower_accepting'] = predictTable['gasprice'].apply(get_hpa, args=(hashpower,))
    predictTable['hashpower_accepting2'] = predictTable['gasprice'].apply(get_hpa, args=(hpower,))
    gp_lookup = predictTable.set_index('gasprice')['hashpower_accepting'].to_dict()
    gp_lookup2 = predictTable.set_index('gasprice')['hashpower_accepting2'].to_dict()

    if not txpool_by_gp.empty:
        predictTable['tx_atabove'] = predictTable['gasprice'].apply(get_tx_atabove, args=(txpool_by_gp,))
        txatabove_lookup = predictTable.set_index('gasprice')['tx_atabove'].to_dict()
    else:
        txatabove_lookup = None
    if not submitted_5mago.empty:
        predictTable['s5mago'] = predictTable['gasprice'].apply(check_recent, args= (submitted_5mago,))
        predictTable['pct_mined_5m'] =  predictTable['gasprice'].apply(get_recent_value, args=(submitted_5mago, 'pct_mined'))
        predictTable['total_seen_5m'] =  predictTable['gasprice'].apply(get_recent_value, args=(submitted_5mago, 'total'))
        s5mago_lookup = predictTable.set_index('gasprice')['s5mago'].to_dict()
    else:
        s5mago_lookup = None
    if not submitted_30mago.empty:
        predictTable['s1hago'] = predictTable['gasprice'].apply(check_recent, args= (submitted_30mago,))
        predictTable['pct_mined_30m'] = predictTable['gasprice'].apply(get_recent_value, args=(submitted_30mago, 'pct_mined'))
        predictTable['total_seen_30m'] = predictTable['gasprice'].apply(get_recent_value, args=(submitted_30mago, 'total'))
        s1hago_lookup = predictTable.set_index('gasprice')['s1hago'].to_dict()
    else:
        s1hago_lookup = None
    predictTable['gas_offered'] = 0
    predictTable['highgas2'] = 0
    predictTable['chained'] = 0
    predictTable['expectedWait'] = predictTable.apply(predict, axis=1)
    predictTable['expectedTime'] = predictTable['expectedWait'].apply(lambda x: np.round((x * avg_timemined / 60), decimals=2))

    return (predictTable, txatabove_lookup, gp_lookup, gp_lookup2)

def get_gasprice_recs(prediction_table, block_time, block, speed, array5m, array30m,  minlow=-1, submitted_5mago=None, submitted_30mago=None):
    """gets gasprice recommendations from txpool and model estimates"""

    def gp_from_txpool(timeframe, calc):
        """calculates the gasprice from the txpool"""
        if timeframe == 'average':
            label_df = ['s5mago', 'pct_mined_5m', 'total_seen_5m']
        elif timeframe == 'safelow':
            label_df = ['s1hago', 'pct_mined_30m', 'total_seen_30m']
        try:
            series = prediction_table.loc[(prediction_table[label_df[0]] <= 5) & (prediction_table[label_df[1]] > 1) & (prediction_table[label_df[2]] > 10), 'gasprice']
            txpool = series.min()
            print ("\ncalc value :" + str(calc))
            print ('txpool value :' + str(txpool))
            if (txpool < calc):
                rec = txpool
            elif (txpool > calc) and (prediction_table.loc[prediction_table['gasprice'] == (calc), label_df[0]].values[0] > 15):
                print ("txpool>calc")
                rec = txpool
            else:
                rec = calc
        except Exception as e:
            txpool = np.nan
            rec = np.nan
        return (rec, txpool)


    def get_safelow():
        series = prediction_table.loc[prediction_table['hashpower_accepting'] >= 35, 'gasprice']
        safelow_calc = series.min()
        (safelow, safelow_txpool) = gp_from_txpool('safelow', safelow_calc)
        if safelow is np.nan:
            safelow = safelow_calc
        minhash_list = prediction_table.loc[prediction_table['hashpower_accepting']>=10, 'gasprice']
        if (safelow < minhash_list.min()):
            safelow = minhash_list.min()
        if minlow >= 0:
            if safelow < minlow:
                safelow = minlow
        safelow = float(safelow)
        safelow_txpool = float(safelow_txpool)
        safelow_calc = float(safelow_calc)
        return (safelow, safelow_calc, safelow_txpool)

    def get_average():
        series = prediction_table.loc[prediction_table['hashpower_accepting'] >= 60, 'gasprice']
        average_calc = series.min()
        (average, average_txpool) = gp_from_txpool('average', average_calc)
        if average is np.nan:
            average = average_calc
        minhash_list = prediction_table.loc[prediction_table['hashpower_accepting']>25, 'gasprice']
        if average < minhash_list.min():
            average = minhash_list.min()
        if np.isnan(average):
            average = average_calc
        average = float(average)
        average_txpool = float(average_txpool)
        average_calc = float(average_calc)
        return (average, average_calc, average_txpool)

    def get_fast():
        series = prediction_table.loc[prediction_table['hashpower_accepting'] >= 90, 'gasprice']
        fastest = series.min()
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

    def check_recent_mediangp (gprec, gparray, calc):
        """stabilizes gp estimates over last 20m"""
        try:
            pct50 = np.percentile(gparray, 50)

            if gprec <= pct50:
                pct50 = pct50/10
                gprec_m = np.ceil(pct50) * 10
            else:
                pct95 = np.percentile(gparray, 95)
                if gprec < pct95:
                    gprec_m = gprec
                else:
                    pct95 = pct95/10
                    gprec_m = np.ceil(pct95) * 10
            if gparray.size > 80:
                gparray = np.delete(gparray, 0)
            if (gprec <= calc) and (gprec_m > calc):
                gprec_m = calc
        except Exception as e:
            print (e)
            gprec_m = gprec
        print ('medianizer: ' + str(gprec_m))
        return (gprec_m, gparray)

    gprecs = {}

    (gprecs['safeLow'], gprecs['safelow_calc'], gprecs['safelow_txpool']) = get_safelow()

    (gprecs['average'], gprecs['average_calc'], gprecs['average_txpool']) = get_average()
    
    gprecs['fast'] = get_fast()

    print("")
    if gprecs['safelow_txpool'] is not np.nan :
        array30m = np.append(array30m, gprecs['safelow_txpool'])
    else:
        array30m = np.append(array30m, gprecs['safeLow'])
    (gprecs['safeLow'], array30m) = check_recent_mediangp(gprecs['safeLow'], array30m, gprecs['safelow_calc'])
    gprecs['safelow_txpool'] = gprecs['safeLow']

    if gprecs['average_txpool'] is not np.nan :
        array5m = np.append(array5m, gprecs['average_txpool'])
    else:
        array5m = np.append(array5m, gprecs['average'])
    (gprecs['average'], array5m) = check_recent_mediangp(gprecs['average'], array5m, gprecs['average_calc'])
    gprecs['average_txpool'] = gprecs['average']

    if (gprecs['fast'] < gprecs['average']):
        gprecs['fast'] = gprecs['average']

    if (gprecs['safeLow'] > gprecs['average']):
        gprecs['safeLow'] = gprecs['average']
        gprecs['safelow_txpool'] = gprecs['average']

    gprecs['safeLowWait'] = get_wait(gprecs['safeLow'])
    gprecs['avgWait'] = get_wait(gprecs['average'])
    
    gprecs['fastWait'] = get_wait(gprecs['fast'])
    gprecs['fastest'] = get_fastest()
    gprecs['fastestWait'] = get_wait(gprecs['fastest'])
    gprecs['block_time'] = block_time
    gprecs['blockNum'] = block
    gprecs['speed'] = speed
    return(gprecs, array5m, array30m)

def make_recent_blockdf(recentdf, current_txpool, alltx):
    '''creates df for monitoring recentlymined tx'''

    def roundresult(row):
        if np.isnan(row[0]) or np.isnan(row[1]):
            return 0
        else:
            x = row[0] / row[1] * 100
            return np.round(x)

    recentdf['still_here'] = recentdf.index.isin(current_txpool.index).astype(int)
    recentdf['mined'] = recentdf.index.isin(alltx.index[alltx['block_mined'].notnull()]).astype(int)
    recentdf = recentdf[['gas_price', 'round_gp_10gwei', 'still_here', 'mined']].groupby('round_gp_10gwei').agg({'gas_price':'count', 'still_here':'sum', 'mined':'sum'})
    recentdf.rename(columns={'gas_price':'total'}, inplace=True)
    recentdf['pct_unmined'] = recentdf[['still_here', 'total']].apply(roundresult, axis=1)
    recentdf['pct_mined'] = recentdf[['mined', 'total']].apply(roundresult, axis=1)
    return recentdf



def analyze_txpool(block, txpool_block, hashpower, hpower, avg_timemined, gaslimit, txatabove_lookup, gp_lookup, gp_lookup2, gprecs):
    """calculate data for transactions in the txpoolblock"""
    txpool_block = txpool_block.loc[txpool_block['block_posted']==block].copy()
    txpool_block['pct_limit'] = txpool_block['gas_offered'].apply(lambda x: x / gaslimit)
    txpool_block['high_gas_offered'] = (txpool_block['pct_limit'] > modelparams.HIGHGAS1).astype(int)
    txpool_block['highgas2'] = (txpool_block['pct_limit'] > modelparams.HIGHGAS2).astype(int)
    txpool_block['hashpower_accepting'] = txpool_block['round_gp_10gwei'].apply(lambda x: gp_lookup[x] if x in gp_lookup else 100)
    txpool_block['hashpower_accepting2'] = txpool_block['round_gp_10gwei'].apply(lambda x: gp_lookup2[x] if x in gp_lookup2 else 100)
    if txatabove_lookup is not None:
        txpool_block['tx_atabove'] = txpool_block['round_gp_10gwei'].apply(lambda x: txatabove_lookup[x] if x in txatabove_lookup else 1)
    txpool_block['expectedWait'] = txpool_block.apply(predict, axis=1)
    txpool_block['expectedTime'] = txpool_block['expectedWait'].apply(lambda x: np.round((x * avg_timemined / 60), decimals=2))
    txpool_block['safelow_calc'] = gprecs['safelow_calc']
    txpool_block['safelow_txpool'] = gprecs['safelow_txpool']
    txpool_block['average_calc'] = gprecs['average_calc']
    txpool_block['average_txpool'] = gprecs['average_txpool']
    return(txpool_block)
