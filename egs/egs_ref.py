"""
egs_ref.py

Utility functions, MySQL Schemas, and other such architecture
for the EthGasStation adaptive oracle.
"""

import pandas as pd
import numpy as np
import json
import urllib
import time
import random
import string
from hexbytes import HexBytes
from sqlalchemy import create_engine, inspect
from .output import Output, OutputException
from .txbatch import TxBatch
from .modelparams.constants import *
from .jsonexporter import JSONExporter, JSONExporterException
from .report_generator import SummaryReport
from .txbatch import TxBatch
import egs.settings
egs.settings.load_settings()
connstr = egs.settings.get_mysql_connstr()
exporter = JSONExporter()
web3 = egs.settings.get_web3_provider()
console = Output()
engine = create_engine(connstr, echo=False, pool_recycle=3600)
conn = engine.connect()

class CleanTx():
    """transaction object / methods for pandas"""
    to_address = None
    from_address = None

    def __init__(self, tx_obj, block_posted=None, time_posted=None, miner=None):
        self.hash = tx_obj.hash
        self.block_posted = block_posted
        self.block_mined = tx_obj.blockNumber
        if 'to' in tx_obj and isinstance(tx_obj['to'], str):
            self.to_address = tx_obj['to'].lower()
        if 'from' in tx_obj and isinstance(tx_obj['from'], str):
            self.from_address = tx_obj['from'].lower()
        self.time_posted = time_posted
        self.gas_price = tx_obj['gasPrice']
        self.gas_offered = tx_obj['gas']
        self.round_gp_10gwei = None
        self.round_gp()
        if isinstance(miner, str):
            self.miner = miner.lower()
        else:
            self.miner = miner
        self.nonce = tx_obj['nonce']

    def to_dataframe(self):
        data = {self.hash: {'block_posted':self.block_posted, 'block_mined':self.block_mined, 'to_address':self.to_address, 'from_address':self.from_address, 'nonce':self.nonce, 'time_posted':self.time_posted, 'time_mined': None, 'gas_price':self.gas_price, 'miner':self.miner, 'gas_offered':self.gas_offered, 'round_gp_10gwei':self.round_gp_10gwei}}
        df = pd.DataFrame.from_dict(data, orient='index')
        return df

    def round_gp(self):
        """Rounds the gas price to gwei"""
        gp = self.gas_price/1e8
        if gp >= 1 and gp < 10:
            gp = int(np.ceil(gp))
        elif gp >= 10:
            gp = gp/10
            gp = int(np.ceil(gp))
            gp = gp*10
        else:
            gp = 0
        self.round_gp_10gwei = gp

class CleanBlock():
    """block object/methods for pandas"""
    def __init__(self, block_obj, main, uncle, timemined, mingasprice=None, numtx = None, weightedgp=None, includedblock=None):
        self.block_number = block_obj.number 
        self.gasused = block_obj.gasUsed
        if 'miner' in block_obj and isinstance(block_obj['miner'], str):
            self.miner = block_obj.miner.lower()
        self.time_mined = timemined
        self.gaslimit = block_obj.gasLimit 
        self.numtx = numtx
        self.blockhash = block_obj.hash.hex()
        self.mingasprice = mingasprice
        self.uncsreported = len(block_obj.uncles)
        self.blockfee = block_obj.gasUsed * weightedgp / 1e10
        self.main = main
        self.uncle = uncle
        self.includedblock = includedblock
        self.speed = self.gasused / self.gaslimit
    
    def to_dataframe(self):
        data = {0:{'block_number':self.block_number, 'gasused':self.gasused, 'miner':self.miner, 'gaslimit':self.gaslimit, 'numtx':self.numtx, 'blockhash':self.blockhash, 'time_mined':self.time_mined, 'mingasprice':self.mingasprice, 'uncsreported':self.uncsreported, 'blockfee':self.blockfee, 'main':self.main, 'uncle':self.uncle, 'speed':self.speed, 'includedblock':self.includedblock}}
        return pd.DataFrame.from_dict(data, orient='index')

def predict(row):
    """predicts block wait time according to model coefs from model params"""
    if row['chained'] == 1:
        return np.nan
    try:
        sum1 = (INTERCEPT + (row['hashpower_accepting'] * HPA_COEF))
        prediction = np.exp(sum1)
        if prediction < 2:
            prediction = 2
        if row['gas_offered'] > 2000000:
            prediction = prediction + 100
        return np.round(prediction, decimals=2)
    except Exception as e:
        console.error("predict: Exception caught: " + str(e))
        return np.nan

def make_gp_index():
    df = pd.DataFrame(index=range(0,1001))
    return (df)

class TxpoolContainer ():
    """Handles txpool dataframe and analysis methods"""
    def __init__(self):
        self.txpool_df = pd.DataFrame() # aggregate list of txhashes in txp
        self.txpool_block = pd.DataFrame() # txp data at block
        self.txpool_by_gp = pd.DataFrame() # txp data grouped by gp
        self.got_txpool = False
    
    def append_current_txp(self):
        """gets list of all txhash in txpool at block and appends to dataframe"""
        hashlist = []
        current_block = web3.eth.blockNumber
        try:
            console.info("getting txpool hashes at block " +str(current_block) + " ...")
            txpoolcontent = web3.txpool.content
            txpoolpending = txpoolcontent['pending']
            for tx_sequence in txpoolpending.values():
                for tx_obj in tx_sequence.values():
                    hashlist.append(tx_obj['hash'])
            txpool_current = pd.DataFrame(index = hashlist)
            txpool_current['block'] = current_block
            console.info("done. length = " +str(len(txpool_current)))
            self.txpool_df = self.txpool_df.append(txpool_current, ignore_index = False)   
        except Exception as e:
            console.warn(e)
            console.warn("txpool empty")
    
    def make_txpool_block(self, block, alltx):
        """gets txhash from all transactions in txpool at block and merges the data from alltx"""

        #get txpool hashes at block
        txpool_block = self.txpool_df.loc[self.txpool_df['block']==block]
        if not txpool_block.empty:
            txpool_block = txpool_block.drop(['block'], axis=1)
            #merge transaction data for txpool transactions
            #txpool_block only has transactions received by filter
            txpool_block = txpool_block.join(alltx, how='inner')
            txpool_block = txpool_block.append(alltx.loc[alltx['block_posted']==block])
            txpool_block = txpool_block.drop_duplicates(keep='first')
            console.info('txpool block length ' + str(len(txpool_block)))
            self.got_txpool = True
        else:
            txpool_block = alltx.loc[alltx['block_posted']==block].copy()
            self.got_txpool = False
            console.info('txpool skipped')
        
        if self.got_txpool:
            self.txpool_block = txpool_block
            #create new df with txpool grouped by gp 
            self.txpool_by_gp = txpool_block[['gas_price', 'round_gp_10gwei']].groupby('round_gp_10gwei').agg({'gas_price':'count'})
            self.txpool_by_gp.rename(columns = {'gas_price':'tx_atabove'}, inplace=True)
            df = make_gp_index()
            temp = self.txpool_by_gp.loc[self.txpool_by_gp.index >=1000]
            if not temp.empty:
                high_tx = temp['tx_atabove'].sum()        
                self.txpool_by_gp.loc[1000, 'tx_atabove'] = high_tx
                self.txpool_by_gp = self.txpool_by_gp.loc[self.txpool_by_gp.index <= 1000]
            self.txpool_by_gp = self.txpool_by_gp.sort_index(ascending=False)
            self.txpool_by_gp = self.txpool_by_gp.cumsum()
            df = df.join(self.txpool_by_gp, how = 'left')
            df = df.fillna(method='bfill')
            df = df.fillna(0)
            self.txpool_by_gp = df[['tx_atabove']].to_dict(orient='index')

            
        else:
            self.txpool_block = pd.DataFrame()
            self.txpool_by_gp = {}
            console.warn("txpool block empty")
    
    def prune(self, block):
        self.txpool_df = self.txpool_df.loc[self.txpool_df['block'] > (block-10)]
        
        
class BlockDataContainer():
    """Handles block-level dataframe and its processing"""
    def __init__(self):
        self.blockdata_df = pd.DataFrame()
        self.block_sumdf = pd.DataFrame()
        self.hashpower = pd.DataFrame()
        self.block_time = None
        self.gaslimit = None
        self.speed = None
        self.init_df()
    
    def init_df(self):
        ins = inspect(engine)
        if 'blockdata2' in ins.get_table_names():
            try:
                self.blockdata_df= pd.read_sql('SELECT * from blockdata2 order by block_number desc limit 2000', con=engine)
            except Exception as e:
                console.warn(e)


    def process_block_data (self, block_transactions_df, block_obj):
        """gets block-level summary data and append to block dataframe"""
        console.debug("Processing block data...")
        if len(block_obj.transactions) > 0:
            block_transactions_df['weighted_fee'] = block_transactions_df['round_gp_10gwei']* block_transactions_df['gas_offered']
            block_mingasprice = block_transactions_df['round_gp_10gwei'].min()
            block_weightedfee = block_transactions_df['weighted_fee'].sum() / block_transactions_df['gas_offered'].sum()
        else:
            block_mingasprice = np.nan
            block_weightedfee = np.nan
        block_numtx = len(block_obj.transactions)
        timemined = block_transactions_df['time_mined'].min()
        clean_block = CleanBlock(block_obj, 1, 0, timemined, block_mingasprice, block_numtx, block_weightedfee).to_dataframe()
        self.block_sumdf = clean_block
        self.blockdata_df = self.blockdata_df.append(clean_block, ignore_index = True)

    def analyze_last200blocks(self, block):
        """analyzes % of last 200 blocks by min mined gas price, summary stats """
        blockdata = self.blockdata_df
        recent_blocks = blockdata.loc[blockdata['block_number'] > (block - 200), ['mingasprice', 'block_number', 'gaslimit', 'time_mined', 'speed']]
        gaslimit = recent_blocks['gaslimit'].mean()
        last10 = recent_blocks.sort_values('block_number', ascending=False).head(n=10)
        speed = last10['speed'].mean()
        #create hashpower accepting dataframe based on mingasprice accepted in block
        df = make_gp_index()
        hashpower = recent_blocks[['mingasprice', 'block_number']].groupby('mingasprice').count()
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
            avg_timemined = 15
        df = df.join(hashpower, how='left')
        df = df.fillna(method = 'ffill')
        df = df.fillna(0)
        self.hashpower = df[['hashp_pct']].astype(int).to_dict(orient='index')
        self.block_time = avg_timemined
        self.gaslimit = gaslimit
        self.speed = speed
        
    def write_to_sql(self):
        """write data to mysql for analysis"""
        self.blockdata_df = self.blockdata_df.sort_values(by=['block_number'], ascending=False)
        self.blockdata_df = self.blockdata_df.head(1500)
        self.blockdata_df.to_sql(con=engine, name='blockdata2', if_exists='replace', index=False)
        console.info("wrote " + str(len(self.blockdata_df)) + " blocks to mysql")

    def prune(self, block):
        """keep dataframes and databases from getting too big"""
        console.info('pruning blockdata')
        deleteBlock = block-5000
        self.blockdata_df = self.blockdata_df.loc[self.blockdata_df['block_number'] > deleteBlock]

class AllTxContainer():
    """Handles transaction dataframe and analysis"""
    def __init__(self):
        self.df = pd.DataFrame()
        self.minedblock_tx_df = pd.DataFrame()
        self.block_obj = None
        self.forced_skips = 0
        self.pending_filter = web3.eth.filter('pending')
        self.load_txdata()
        self.process_block = web3.eth.blockNumber
        self.new_tx_list = []
        self.pctmined_gp_last100 = pd.DataFrame()
        
    
    def load_txdata(self):
        """load data from mysql into dataframes"""
        try:
            ins = inspect(engine)
            if 'alltx' in ins.get_table_names():
                alltx = pd.read_sql('SELECT * from alltx order by block_posted desc limit 100000', con=engine)
                alltx.set_index('index', drop=True, inplace=True)
                if 'level_0' in alltx.columns:
                    self.df = alltx.drop('level_0', axis=1)
            else:
                return
        except Exception as e:
            console.warn(e)


    def listen(self):
        """listens for new pending tx and adds them to the alltx dataframe"""
        #Set number of transactions to sample to keep from falling behind; can be adjusted
        current_block = web3.eth.blockNumber
        console.info ("listening for new transactions at block "+ str(current_block)+"...." )
        self.new_tx_list = []
        try:
            while True:
                if self.process_block < (current_block - 5):
                    console.warn("blocks jumped, skipping ahead")
                    self.process_block = current_block
                    self.forced_skips = self.forced_skips + 1

                try:
                    # console.debug("Getting filter changes...")
                    self.new_tx_list.extend(self.pending_filter.get_new_entries())
                except:
                    # filters suck. The node can kill them whenever it wants.
                    console.warn("Pending transaction filter missing, re-establishing filter")
                    self.pending_filter = web3.eth.filter('pending')
                    self.new_tx_list.extend(self.pending_filter.get_new_entries())

                current_block = web3.eth.blockNumber
    
                if self.process_block < current_block:
                    console.info('now processing block ' + str(self.process_block))
                    #get unique txids
                    self.new_tx_list = set(self.new_tx_list)
                    return
                else:
                    time.sleep(0.5)
        except Exception as e:
            console.warn(e)

                
    def process_submitted_block(self):
        
        tx_hashes = []

        def getbatch(tx_hashes):
            """get tx objects and account nonces"""
            submitted_block = pd.DataFrame()
            txs = TxBatch(web3)
            try: 
                results = txs.batchRequest('eth_getTransactionByHash', tx_hashes)
                for txhash, txobject in results.items():
                    if txobject is not None:
                        clean_tx = CleanTx(txobject, self.process_block, None)
                        submitted_block = submitted_block.append(clean_tx.to_dataframe(), ignore_index = False)
            except Exception as e:
                raise e
                console.error("Batch transaction failed.")
            
            if len(submitted_block):
                from_addresses = list(set(submitted_block['from_address'].tolist()))
                nonces = TxBatch(web3)
                results = nonces.nonceBatch('eth_getTransactionCount', from_addresses, self.process_block)
                submitted_block['account_nonce'] = submitted_block['from_address'].apply(lambda x: results[x] if x in results else np.nan)
                submitted_block['chained'] = (submitted_block['nonce'] > submitted_block['account_nonce']).astype(int)
                console.info("added tx: " + str(len(submitted_block)))
                self.df = self.df.append(submitted_block)

        '''need to loop through hexbytes list and convert to hexstrings'''
        for txHash in self.new_tx_list:
            if isinstance(txHash, HexBytes):
                tx_hashes.append(txHash.hex().lower())
            elif isinstance(txHash, str):
                tx_hashes.append(txHash.lower())
            else:
                raise TypeError("TxBatch.addTxHash: txHash is not a string or HexBytes")

        '''dont add duplicate transactions'''
        tx_hashes = set(tx_hashes)
        existing = self.df.index.tolist()
        tx_hashes = list(tx_hashes.difference(existing))
        console.info("submitted tx: " + str(len(tx_hashes)))

        '''don't break the pipe'''
        if len(tx_hashes) > 500:
            if len(tx_hashes) > 1000:
                tx_hashes = random.sample(tx_hashes, 1000)
            mid = int(len(tx_hashes)/2)
            tx_hash_sub = [tx_hashes[0:mid-1], tx_hashes[mid:]]
            for hash_list in tx_hash_sub:
                getbatch(hash_list)

        else:
            getbatch(tx_hashes)
    
    def process_mined_transactions(self):
        """get all mined transactions at block and update alltx df"""
        block_df = pd.DataFrame()
        mined_block_num = self.process_block - 3
        block_obj = web3.eth.getBlock(mined_block_num, True)
        miner = block_obj.miner
        for transaction in block_obj.transactions:
            clean_tx = CleanTx(transaction, None, None, miner)
            clean_tx.hash = clean_tx.hash.hex()
            block_df = block_df.append(clean_tx.to_dataframe(), ignore_index = False)
        block_df['time_mined'] = block_obj.timestamp

        #add mined data to dataframe
        mined_blockdf_seen = block_df[block_df.index.isin(self.df.index)]
        console.info('num mined in ' + str(mined_block_num)+ ' = ' + str(len(block_df)))
        console.info('num seen in ' + str(mined_block_num)+ ' = ' + str(len(mined_blockdf_seen)))
        #update transactions with mined data
        self.df = self.df.combine_first(block_df)
        self.block_obj = block_obj
        self.minedblock_tx_df = block_df
    
    def analyzetx_last100blocks(self):
        """finds % of transactions mined at gas price over last 100 blocks"""
        alltx = self.df
        block = self.process_block
        recent_blocks = alltx.loc[alltx['block_mined'] > (block-100), ['block_mined', 'round_gp_10gwei']].copy()
        hpower = recent_blocks.groupby('round_gp_10gwei').count()
        hpower = hpower.rename(columns={'block_mined':'count'})
        totaltx  = len(recent_blocks)
        hpower['cum_tx'] = hpower['count'].cumsum()
        hpower['hashp_pct'] = hpower['cum_tx']/totaltx*100
        df = make_gp_index()
        df = df.join(hpower, how='left')
        df = df.fillna(method = 'ffill')
        df = df.fillna(0)
        self.pctmined_gp_last100 = df[['hashp_pct']].astype(int).to_dict(orient='index')
    
    def update_txblock(self, txpool_block, blockdata, predictiontable, gprecs):
        '''
        updates transactions at block with calc values from prediction table
        '''
        if txpool_block is None:
            return
        block = self.process_block
        gaslimit = blockdata.gaslimit
        gp_lookup = predictiontable.gp_lookup
        gp_lookup2 = predictiontable.gp_lookup2
        txatabove_lookup = predictiontable.txatabove_lookup
        recent_lookup = predictiontable.recent_lookup
        remote_lookup = predictiontable.remote_lookup

        txpool_block = txpool_block.loc[txpool_block['block_posted']==block].copy()
        txpool_block['pct_limit'] = txpool_block['gas_offered'].apply(lambda x: x / gaslimit)
        txpool_block['high_gas_offered'] = (txpool_block['pct_limit'] > HIGHGAS1).astype(int)
        txpool_block['highgas2'] = (txpool_block['pct_limit'] > HIGHGAS2).astype(int)
        txpool_block['hashpower_accepting'] = txpool_block['round_gp_10gwei'].apply(lambda x: gp_lookup[x] if x in gp_lookup else 100)
        txpool_block['hashpower_accepting2'] = txpool_block['round_gp_10gwei'].apply(lambda x: gp_lookup2[x] if x in gp_lookup2 else 100)
        if txatabove_lookup is not None:
            txpool_block['tx_atabove'] = txpool_block['round_gp_10gwei'].apply(lambda x: txatabove_lookup[x] if x in txatabove_lookup else 1)
        if recent_lookup is not None:
            txpool_block['s5mago'] = txpool_block['round_gp_10gwei'].apply(lambda x: recent_lookup[x] if x in recent_lookup else 0)
        if remote_lookup is not None:
            txpool_block['s30mago'] = txpool_block['round_gp_10gwei'].apply(lambda x: remote_lookup[x] if x in remote_lookup else 0)
        txpool_block['expectedWait'] = txpool_block.apply(predict, axis=1)
        txpool_block['expectedTime'] = txpool_block['expectedWait'].apply(lambda x: np.round((x * blockdata.block_time / 60), decimals=2))
        txpool_block['safelow_calc'] = gprecs['safelow_calc']
        txpool_block['safelow_txpool'] = gprecs['safelow_txpool']
        txpool_block['average_calc'] = gprecs['average_calc']
        txpool_block['average_txpool'] = gprecs['average_txpool']
        console.info("updating " + str(len(txpool_block)) + " transactions")
        self.df = self.df.combine_first(txpool_block)
        

    def write_to_sql(self):
        """writes to sql, prevent buffer overflow errors"""
        console.info("writing to mysql....this can take awhile")
        self.df.reset_index(inplace=True)
        length = len(self.df)
        chunks = int(np.ceil(length/1000))
        if length < 1000:
            self.df.to_sql(con=engine, name='alltx', if_exists='replace')
        else:
            start = 0
            stop = 999
            for chunck in range(0,chunks):
                tempdf = self.df[start:stop]
                if chunck == 0: 
                    tempdf.to_sql(con=engine, name='alltx', if_exists='replace')
                else:
                    tempdf.to_sql(con=engine, name='alltx', if_exists='append')
                start += 1000
                stop += 1000
                if stop > length:
                    stop = length-1
        console.info("wrote " + str(length) + " transactions to alltx.")

    
    def prune(self):
        """keep dataframes and databases from getting too big"""
        deleteBlock_mined = self.process_block - 1500
        deleteBlock_posted = self.process_block - 4500
        self.df = self.df.loc[(self.df['block_posted'] > deleteBlock_posted)]
        self.df = self.df.loc[(self.df['block_mined'] > deleteBlock_mined)]
    

class RecentlySubmittedTxDf():
    """Df for holding recently submitted tx to track clearing from txpool"""
    def __init__(self, name, current_block, start_block, end_block, max_gas, alltx, txpool):
        self.df = pd.DataFrame()
        self.current_block = current_block
        self.name = name
        self.total_tx = None
        self.start_block = start_block
        self.end_block = end_block
        self.max_gas = max_gas
        self.alltx = alltx
        self.txpool = txpool
        self.init_df()
    
    def init_df(self):
        alltx = self.alltx
        current_txpool = self.txpool.txpool_block

        #get tx matching selection criteria: block submitted, eligible nonce, exclude very high gas offered

        recentdf = alltx.loc[(alltx['block_posted'] > (self.current_block - self.end_block)) & (alltx['block_posted'] < (self.current_block - self.start_block)) & (alltx['chained']==0) & (alltx['gas_offered'] < self.max_gas)].copy()
        self.total_tx = len(recentdf)

        def roundresult(row):
            if np.isnan(row[0]) or np.isnan(row[1]):
                return 0
            else:
                x = row[0] / row[1] * 100
                return np.round(x)
        if (len(recentdf) > 50) & (self.txpool.got_txpool): #only useful if both have sufficient transactions for analysis; otherwise set to empty
            recentdf['still_here'] = recentdf.index.isin(current_txpool.index).astype(int)
            recentdf['mined'] = recentdf.index.isin(alltx.index[alltx['block_mined'].notnull()]).astype(int)
            recentdf['round_gp_10gwei'] = recentdf['round_gp_10gwei'].astype(int)
            recentdf = recentdf[['gas_price', 'round_gp_10gwei', 'still_here', 'mined']].groupby('round_gp_10gwei').agg({'gas_price':'count', 'still_here':'sum', 'mined':'sum'})
            recentdf.rename(columns={'gas_price':'total'}, inplace=True)
            recentdf['pct_unmined'] = recentdf[['still_here', 'total']].apply(roundresult, axis=1)
            recentdf['pct_mined'] = recentdf[['mined', 'total']].apply(roundresult, axis=1)
            self.print_length()
            self.df = recentdf
        else:
            self.print_length()
            self.df = pd.DataFrame()
    
    def print_length(self):
            console.info("# of tx submitted ~ " + str(self.name) + " = " + str(self.total_tx))


class PredictionTable():
    def __init__(self, blockdata, alltx, txpool, recentdf, remotedf):
        self.blockdata = blockdata
        self.alltx = alltx
        self.txpool = txpool
        self.recentdf = recentdf
        self.remotedf = remotedf
        self.predictiondf = pd.DataFrame()
        self.txatabove_lookup = None
        self.gp_lookup = None
        self.gp_lookup2 = None
        self.recent_lookup = None
        self.remote_lookup = None
        self.init_predictiontable()
    
    def init_predictiontable(self):
        """makes prediction table for number of blocks to confirmation"""
        hashpower = self.blockdata.hashpower
        hpower = self.alltx.pctmined_gp_last100
        avg_timemined = self.blockdata.block_time
        txpool_by_gp = self.txpool.txpool_by_gp
        submitted_5mago = self.recentdf
        submitted_30mago = self.remotedf

        def get_recent_value(gasprice, submitted_recent, col):
            """gets values from recenttx df for prediction table"""
            if gasprice in submitted_recent.index:
                rval = submitted_recent.at[gasprice, col]
            else:
                rval = 0
            return rval
        
        def check_recent(gasprice, submitted_recent):
            """gets the %of transactions unmined submitted in recent blocks"""

            #set this to avoid false positive delays
            submitted_recent.loc[(submitted_recent['still_here'] >= 1) & (submitted_recent['still_here'] <= 2) & (submitted_recent['total'] < 4), 'pct_unmined'] = np.nan
            maxval = submitted_recent.loc[submitted_recent.index > gasprice, 'pct_unmined'].max()
            if gasprice in submitted_recent.index:
                stillh = submitted_recent.at[gasprice, 'still_here']
                if stillh > 2:
                    rval =  submitted_recent.at[gasprice, 'pct_unmined']
                else:
                    rval = maxval
            else:
                rval = maxval
            if gasprice >= 1000:
                rval = 0
            if (rval > maxval) or (gasprice >= 1000) :
                return rval
            return maxval


        predictTable = make_gp_index()
        predictTable['hashpower_accepting'] = pd.DataFrame.from_dict(hashpower, orient='index')
        predictTable['hashpower_accepting2'] = pd.DataFrame.from_dict(hpower, orient='index')
        predictTable['tx_atabove'] = pd.DataFrame.from_dict(txpool_by_gp, orient='index')
        
        print (predictTable)
        xxxx
        
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

        self.predictiondf = predictTable
        self.txatabove_lookup = txatabove_lookup
        self.gp_lookup = gp_lookup
        self.gp_lookup2 = gp_lookup2
        self.recent_lookup = s5mago_lookup
        self.remote_lookup = s1hago_lookup

    def write_to_json(self, txpool):
        """write json data unless txpool block empty"""
        global exporter
        try:
            if not txpool.txpool_block.empty:
                self.predictiondf['gasprice'] = self.predictiondf['gasprice']/10
                prediction_tableout = self.predictiondf.to_json(orient='records')
                exporter.write_json('predictTable', prediction_tableout)
        except Exception as e:
            console.error("write_to_json: Exception caught: " + str(e))


class GasPriceReport():
    def __init__(self, predictiontable, blockdata, submitted_recent, submmited_remote, array5m, array30m, block):
        self.predictiontable = predictiontable
        self.blockdata = blockdata
        self.submitted_recent = submitted_recent
        self.submitted_remote = submmited_remote
        self.block = block
        self.array5m = array5m
        self.array30m = array30m
        self.gprecs = None
        self.minlow = 0.1
        self.make_gasprice_report()
        
      
    def make_gasprice_report(self):
        """processes block data and makes the gas price recommendations"""
        prediction_table = self.predictiontable
        block_time = self.blockdata.block_time
        speed = self.blockdata.speed
        array5m = self.array5m
        array30m = self.array30m
        minlow = self.minlow
        block = self.block
        

        def gp_from_txpool(timeframe, calc):
            """calculates the gasprice from the txpool"""
            if timeframe == 'average':
                label_df = ['s5mago', 'pct_mined_5m', 'total_seen_5m']
            elif timeframe == 'safelow':
                label_df = ['s1hago', 'pct_mined_30m', 'total_seen_30m']
        
            if label_df[0] in prediction_table.columns:
                try:
                    #pct_unmined <10%, must have 1% mined (as opposed to dropped), and must have seen at least 2 transactions at the gas price
                    series = prediction_table.loc[(prediction_table[label_df[0]] < 10) & (prediction_table[label_df[1]] > 1) & (prediction_table[label_df[2]] >= 2), 'gasprice']
                    txpool = series.min()
                    console.info("calc value: " + str(calc))
                    console.info("txpool value: " + str(txpool))
                    if (txpool < calc):
                        rec = txpool
                    elif (txpool > calc) and (prediction_table.loc[prediction_table['gasprice'] == (calc), label_df[0]].values[0] > 15):
                        console.warn("txpool > calc")
                        rec = txpool
                    else:
                        rec = calc
                except Exception as e:
                    console.error(e)
                    txpool = np.nan
                    rec = np.nan
                return (rec, txpool)
            else:
                return (np.nan, np.nan)


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
                console.debug("minlow " +str(minlow))
                if safelow < minlow:
                    safelow = minlow
            if safelow < 1:
                safelow = 1
                safelow_txpool = 1
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
                    if pct50 < 10:
                        gprec_m = np.round(pct50)
                    else:
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
                console.error("check_recent_mediagp: Exception caught: " + str(e))
                gprec_m = gprec
            console.info("medianizer: %s" % str(gprec_m))
            return (gprec_m, gparray)

        gprecs = {}

        (gprecs['safeLow'], gprecs['safelow_calc'], gprecs['safelow_txpool']) = get_safelow()

        (gprecs['average'], gprecs['average_calc'], gprecs['average_txpool']) = get_average()

        gprecs['fast'] = get_fast()

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
        self.gprecs = gprecs
        self.array5m = array5m
        self.array30m = array30m


    def write_to_json(self):
        """write json data"""
        global exporter
        try:
            exporter.write_json('ethgasAPI', self.gprecs)
        except Exception as e:
            console.error("write_to_json: Exception caught: " + str(e))

