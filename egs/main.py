"""
ETH Gas Station
Main Event Loop
"""

import time
import sys
import json
import math
import traceback
import os
import random
import pandas as pd
import numpy as np

import egs.settings
egs.settings.load_settings()

from sqlalchemy import create_engine, Column, Integer, String, DECIMAL, BigInteger, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from .egs_ref import *
from .jsonexporter import JSONExporter, JSONExporterException
from .report_generator import SummaryReport
from .per_block_analysis import *
from .output import Output, OutputException


# configure necessary services
exporter = JSONExporter()
web3 = egs.settings.get_web3_provider()
connstr = egs.settings.get_mysql_connstr()
engine = create_engine(connstr, echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
console = Output()


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

def write_to_sql(alltx, block_sumdf, mined_blockdf, block):
    """write data to mysql for analysis"""
    post = alltx[alltx.index.isin(mined_blockdf.index)]
    post.to_sql(con=engine, name='minedtx2', if_exists='append', index=True)
    console.info('num mined = ' + str(len(post)))
    post2 = alltx.loc[alltx['block_posted'] == (block)]
    post2.to_sql(con=engine, name='postedtx2', if_exists='append', index=True)
    console.info('num posted = ' + str(len(post2)))
    block_sumdf.to_sql(con=engine, name='blockdata2', if_exists='append', index=False)

def write_report(report, top_miners, price_wait, miner_txdata, gasguzz, lowprice):
    """write json data"""
    global exporter

    parentdir = os.path.dirname(os.getcwd())
    top_minersout = top_miners.to_json(orient='records')
    minerout = miner_txdata.to_json(orient='records')
    gasguzzout = gasguzz.to_json(orient='records')
    lowpriceout = lowprice.to_json(orient='records')
    price_waitout = price_wait.to_json(orient='records')

    try:
        exporter.write_json('txDataLast10k', report)
        exporter.write_json('topMiners', top_minersout)
        exporter.write_json('priceWait', price_waitout)
        exporter.write_json('miners', minerout)
        exporter.write_json('gasguzz', gasguzzout)
        exporter.write_json('validated', lowpriceout)
    except Exception as e:
        console.error("write_report: Exception caught: " + str(e))

def write_to_json(gprecs, prediction_table=pd.DataFrame()):
    """write json data"""
    global exporter
    try:
        parentdir = os.path.dirname(os.getcwd())
        if not prediction_table.empty:
            prediction_table['gasprice'] = prediction_table['gasprice']/10
            prediction_tableout = prediction_table.to_json(orient='records')
            exporter.write_json('predictTable', prediction_tableout)

        exporter.write_json('ethgasAPI', gprecs)
    except Exception as e:
        console.error("write_to_json: Exception caught: " + str(e))

def master_control(args):
    report_option = False
    if args.generate_report is True:
        report_option = True

    (blockdata, alltx) = init_dfs()
    txpool = pd.DataFrame()
    snapstore = pd.DataFrame()
    console.info('blocks '+ str(len(blockdata)))
    console.info('txcount '+ str(len(alltx)))
    timer = Timers(web3.eth.blockNumber)
    start_time = time.time()
    first_cycle = True
    analyzed = 0


    def append_new_tx(clean_tx):
        nonlocal alltx
        if not clean_tx.hash in alltx.index:
            alltx = alltx.append(clean_tx.to_dataframe(), ignore_index = False)

    def update_dataframes(block):
        nonlocal alltx
        nonlocal txpool
        nonlocal blockdata
        nonlocal timer
        got_txpool = 1

        console.info('updating dataframes at block '+ str(block))
        try:
            #get minedtransactions and blockdata from previous block
            mined_block_num = block-3
            (mined_blockdf, block_obj) = process_block_transactions(mined_block_num)

            #add mined data to tx dataframe
            mined_blockdf_seen = mined_blockdf[mined_blockdf.index.isin(alltx.index)]
            console.info('num mined in ' + str(mined_block_num)+ ' = ' + str(len(mined_blockdf)))
            console.info('num seen in ' + str(mined_block_num)+ ' = ' + str(len(mined_blockdf_seen)))
            alltx = alltx.combine_first(mined_blockdf)

            #process block data
            console.debug("Processing block data...")
            block_sumdf = process_block_data(mined_blockdf, block_obj)

            #add block data to block dataframe
            blockdata = blockdata.append(block_sumdf, ignore_index = True)

            #get hashpower table, block interval time, gaslimit, speed from last 200 blocks
            (hashpower, block_time, gaslimit, speed) = analyze_last200blocks(block, blockdata)
            hpower2 = analyze_last100blocks(block, alltx)

            submitted_30mago = alltx.loc[(alltx['block_posted'] < (block-50)) & (alltx['block_posted'] > (block-120)) & (alltx['chained']==0) & (alltx['gas_offered'] < 500000)].copy()
            console.info("# of tx submitted ~ an hour ago: " + str((len(submitted_30mago))))

            submitted_5mago = alltx.loc[(alltx['block_posted'] < (block-8)) & (alltx['block_posted'] > (block-49)) & (alltx['chained']==0) & (alltx['gas_offered'] < 500000)].copy()
            console.info("# of tx submitted ~ 5m ago: " + str((len(submitted_5mago))))

            if ((len(submitted_30mago) > 50) & (len(current_txpool) > 100)):
                submitted_30mago = make_recent_blockdf(submitted_30mago, current_txpool, alltx)
            else:
                submitted_30mago = pd.DataFrame()

            if ((len(submitted_5mago) > 50) & (len(current_txpool)> 100)):
                submitted_5mago = make_recent_blockdf(submitted_5mago, current_txpool, alltx)
            else:
                submitted_5mago = pd.DataFrame()

            #make txpool block data
            txpool_block = make_txpool_block(block, txpool, alltx)

            if not txpool_block.empty:
                #new dfs grouped by gasprice and nonce
                txpool_by_gp = txpool_block[['gas_price', 'round_gp_10gwei']].groupby('round_gp_10gwei').agg({'gas_price':'count'})
                txpool_block_nonce = txpool_block[['from_address', 'nonce']].groupby('from_address').agg({'nonce':'min'})
                txpool_block = analyze_nonce(txpool_block, txpool_block_nonce)
            else:
                txpool_by_gp = pd.DataFrame()
                txpool_block_nonce = pd.DataFrame()
                txpool_block = alltx.loc[alltx['block_posted']==block]
                got_txpool = 0

            #make prediction table and create lookups to speed txpool analysis
            (predictiondf, txatabove_lookup, gp_lookup, gp_lookup2) = make_predcitiontable(hashpower, hpower2, block_time, txpool_by_gp, submitted_5mago, submitted_30mago)

            #with pd.option_context('display.max_rows', None,):
                #print(predictiondf)

            #make the gas price recommendations
            (gprecs, timer.gp_avg_store, timer.gp_safelow_store) = get_gasprice_recs (predictiondf, block_time, block, speed, timer.gp_avg_store, timer.gp_safelow_store, timer.minlow, submitted_5mago, submitted_30mago)

            #create the txpool block data
            #first, add txs submitted if empty

            try:
                if txpool_block.notnull:
                    analyzed_block = analyze_txpool(block, txpool_block, hashpower, hpower2, block_time, gaslimit, txatabove_lookup, gp_lookup, gp_lookup2, gprecs)
                    #update alltx
                    alltx = alltx.combine_first(analyzed_block)
            except:
                pass

            #with pd.option_context('display.max_columns', None,):
                #print(analyzed_block)

            #make summary report every x blocks
            #this is only run if generating reports for website
            if report_option is True:
                if timer.check_reportblock(block):
                    last1500t = alltx[alltx['block_mined'] > (block-1500)].copy()
                    console.info('txs '+ str(len(last1500t)))
                    last1500b = blockdata[blockdata['block_number'] > (block-1500)].copy()
                    console.info('blocks ' +  str(len(last1500b)))
                    report = SummaryReport(last1500t, last1500b, block)
                    console.debug("Writing summary reports for web...")
                    write_report(report.post, report.top_miners, report.price_wait, report.miner_txdata, report.gasguzz, report.lowprice)
                    timer.minlow = report.minlow


            #every block, write gprecs, predictions, txpool by gasprice

            if got_txpool:
                write_to_json(gprecs, predictiondf)
            else:
                write_to_json(gprecs)

            console.debug("Writing transactions to SQL database")
            write_to_sql(alltx, block_sumdf, mined_blockdf, block)

            #keep from getting too large
            console.debug("Pruning database")
            (blockdata, alltx, txpool) = prune_data(blockdata, alltx, txpool, block)
            return True

        except:
            console.error(traceback.format_exc())


    while True:
        try:
            block = web3.eth.blockNumber
            if first_cycle == True and block != analyzed:
                analyzed = block
                tx_filter = web3.eth.filter('pending')
                #get list of txhashes from txpool
                console.info("getting txpool hashes at block " +str(block) +" ...")
                current_txpool = get_txhashes_from_txpool(block)
                #add txhashes to txpool dataframe
                console.info("done. length = " +str(len(current_txpool)))
                txpool = txpool.append(current_txpool, ignore_index = False)
        except:
            pass

        try:
            #console.debug("Getting filter changes...")
            new_tx_list = web3.eth.getFilterChanges(tx_filter.filter_id)
        except:
            console.warn("pending filter missing, re-establishing filter")
            tx_filter = web3.eth.filter('pending')
            new_tx_list = web3.eth.getFilterChanges(tx_filter.filter_id)

        timestamp = time.time()

        #this can be adjusted depending on how fast your server is
        if timer.process_block <= (block-5) and len(new_tx_list) > 10:
            console.info("sampling 10 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 10)
        elif timer.process_block == (block-4) and len(new_tx_list) > 25:
            console.info("sampling 25 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 25)
        elif timer.process_block == (block-3) and len(new_tx_list) > 50:
            console.info("sampling 50 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 50)
        elif timer.process_block == (block-2) and len(new_tx_list) > 75:
            console.info("sampling 75 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 75)
        elif timer.process_block == (block-1) and len(new_tx_list) > 100:
            console.info("sampling 100 from " + str(len(new_tx_list)) + " new tx")
            new_tx_list = random.sample(new_tx_list, 100)

        #if new_tx_list:
            #console.debug("Analyzing %d new transactions from txpool." % len(new_tx_list))
        for new_tx in new_tx_list:
            try:
                #console.debug("Get Tx %s" % new_tx)
                tx_obj = web3.eth.getTransaction(new_tx)
                clean_tx = CleanTx(tx_obj, block, timestamp)
                clean_tx.to_address = clean_tx.to_address.lower()
                append_new_tx(clean_tx)
            except Exception as e:
                console.debug("Exception on Tx %s" % new_tx)

        first_cycle = False

        if (timer.process_block < block):
            try:
                test_filter = web3.eth.uninstallFilter(tx_filter.filter_id)
            except:
                pass
            console.info('current block ' +str(block))
            console.info('processing block ' + str(timer.process_block))
            updated = update_dataframes(timer.process_block)
            console.info('finished ' + str(timer.process_block))
            timer.process_block = timer.process_block + 1
            first_cycle = True

        if (timer.process_block < (block - 8)):
            console.warn("blocks jumped, skipping ahead")
            timer.process_block = (block-1)