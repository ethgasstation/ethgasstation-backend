import time
import sys
import json
import math
import traceback
import os
import random
import pandas as pd
import numpy as np
from web3 import Web3, HTTPProvider
from sqlalchemy import create_engine, Column, Integer, String, DECIMAL, BigInteger, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from egs import *
from per_block_analysis import *
from report_generator import *

web3 = Web3(HTTPProvider('http://localhost:8545'))
engine = create_engine(
    'mysql+mysqlconnector://ethgas:station@127.0.0.1:3306/tx', echo=False)
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
    deleteBlock_sql = block - 3500
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

def write_report(report, top_miners, price_wait, miner_txdata, gasguzz, lowprice):
    """write json data"""
    parentdir = os.path.dirname(os.getcwd())
    top_minersout = top_miners.to_json(orient='records')
    minerout = miner_txdata.to_json(orient='records')
    gasguzzout = gasguzz.to_json(orient='records')
    lowpriceout = lowprice.to_json(orient='records')
    price_waitout = price_wait.to_json(orient='records')
    filepath_report = parentdir + '/json/txDataLast10k.json'
    filepath_tminers = parentdir + '/json/topMiners.json'
    filepath_pwait = parentdir + '/json/priceWait.json'
    filepath_minerout = parentdir + '/json/miners.json'
    filepath_gasguzzout = parentdir + '/json/gasguzz.json'
    filepath_lowpriceout = parentdir + '/json/validated.json'

    try:
        with open(filepath_report, 'w') as outfile:
            json.dump(report, outfile, allow_nan=False)
        with open(filepath_tminers, 'w') as outfile:
            outfile.write(top_minersout)
        with open(filepath_pwait, 'w') as outfile:
            outfile.write(price_waitout)
        with open(filepath_minerout, 'w') as outfile:
            outfile.write(minerout)
        with open(filepath_gasguzzout, 'w') as outfile:
            outfile.write(gasguzzout)
        with open(filepath_lowpriceout, 'w') as outfile:
            outfile.write(lowpriceout)

    except Exception as e:
        print(e)

def write_to_json(gprecs, txpool_by_gp, prediction_table, analyzed_block, submitted_hourago=None):
    """write json data"""
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
        parentdir = os.path.dirname(os.getcwd())
        filepath_gprecs = parentdir + '/json/ethgasAPI.json'
        filepath_txpool_gp = parentdir + '/json/memPool.json'
        filepath_prediction_table = parentdir + '/json/predictTable.json'
        filepath_analyzedblock = parentdir + '/json/txpoolblock.json'
        
        with open(filepath_gprecs, 'w') as outfile:
            json.dump(gprecs, outfile)

        with open(filepath_prediction_table, 'w') as outfile:
            outfile.write(prediction_tableout)

        with open(filepath_txpool_gp, 'w') as outfile:
            outfile.write(txpool_by_gpout)

        with open(filepath_analyzedblock, 'w') as outfile:
            outfile.write(analyzed_blockout)

        if not submitted_hourago.empty:
            submitted_hourago = submitted_hourago.to_json(orient='records')
            filepath_hourago = parentdir + '/json/hourago.json'
            with open(filepath_hourago, 'w') as outfile:
                outfile.write(submitted_hourago)

    
    except Exception as e:
        print(e)
    
def master_control(report_option):
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
            current_txpool = get_txhases_from_txpool(block)

            #add txhashes to txpool dataframe
            txpool = txpool.append(current_txpool, ignore_index = False)

            #get hashpower table, block interval time, gaslimit, speed from last 200 blocks
            (hashpower, block_time, gaslimit, speed) = analyze_last200blocks(block, blockdata)
            hpower2 = analyze_last100blocks(block, alltx)

            submitted_hourago = alltx.loc[(alltx['block_posted'] < (block-130)) & (alltx['block_posted'] > (block-260)) & (alltx['chained']==0) & (alltx['gas_offered'] < 500000)].copy()
            print("# of tx submitted ~ an hour ago: " + str((len(submitted_hourago))))

            submitted_5mago = alltx.loc[(alltx['block_posted'] < (block-20)) & (alltx['block_posted'] > (block-70)) & (alltx['chained']==0) & (alltx['gas_offered'] < 500000)].copy()
            print("# of tx submitted ~ 5m ago: " + str((len(submitted_5mago))))

            if len(submitted_hourago > 50):
                submitted_hourago = make_recent_blockdf(submitted_hourago, current_txpool)
            else:
                submitted_hourago = pd.DataFrame()

            if len(submitted_5mago > 50):
                submitted_5mago = make_recent_blockdf(submitted_5mago, current_txpool)
            else:
                submitted_5mago = pd.DataFrame()

            #make txpool block data
            (analyzed_block, txpool_by_gp, predictiondf) = analyze_txpool(block-1, txpool, alltx, hashpower, block_time, gaslimit, submitted_5mago, submitted_hourago, hpower2)
            if analyzed_block.empty:
                print("txpool block is empty - returning")
                return
            assert analyzed_block.index.duplicated().sum()==0
            alltx = alltx.combine_first(analyzed_block)

            

            #with pd.option_context('display.max_columns', None,):
                #print(analyzed_block)

            #get gpRecs
            gprecs = get_gasprice_recs (predictiondf, block_time, block, speed, timer.minlow, submitted_hourago)

            #make summary report every x blocks
            #this is only run if generating reports for website
            if report_option == '-r':
                if timer.check_reportblock(block):
                    last1500t = alltx[alltx['block_mined'] > (block-1500)].copy()
                    print('txs '+ str(len(last1500t)))
                    last1500b = blockdata[blockdata['block_number'] > (block-1500)].copy()
                    print('blocks ' +  str(len(last1500b)))
                    report = SummaryReport(last1500t, last1500b, block)
                    write_report(report.post, report.top_miners, report.price_wait, report.miner_txdata, report.gasguzz, report.lowprice)
                    timer.minlow = report.minlow


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

        #this can be adjusted depending on how fast your server is
        if timer.process_block <= (block-5) and len(new_tx_list) > 10:
            print("sampling 10 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 10)
        elif timer.process_block == (block-4) and len(new_tx_list) > 25:
            print("sampling 25 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 25)
        elif timer.process_block == (block-3) and len(new_tx_list) > 50:
            print("sampling 50 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 50)
        elif timer.process_block == (block-2) and len(new_tx_list) > 100:
            print("sampling 100 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 100)
        elif timer.process_block == (block-1) and len(new_tx_list) > 200:
            print("sampling 200 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 200)
       
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
                print ('finished ' + str(timer.process_block) + "\n")
                timer.process_block = timer.process_block + 1
        
        if (timer.process_block < (block - 8)):
                print("skipping ahead \n")
                timer.process_block = (block-1)
              
    
            
if len(sys.argv) > 1:            
    report_option = sys.argv[1] # '-r' = make website report
else:
    report_option = False

master_control(report_option)
