"""
ETH Gas Station
Main Event Loop
"""

import traceback
from .egs_ref import *
from .output import Output, OutputException

console = Output()

def master_control(args):
    report_option = False
    if args.generate_report is True:
        report_option = True

    blockdata = BlockDataContainer()
    alltx = AllTxContainer()
    txpool = TxpoolContainer()
    array5m = []
    array30m = []
    console.info('blocks '+ str(len(blockdata.df)))
    console.info('txcount '+ str(len(alltx.df)))

    start_time = time.time()
    analyzed = 0
    last_prune = None

    while True:
        try:
            #get the hashes in the current txpool
            txpool.append_current_txp() 
            #add new pending transactions until new block arrives
            alltx.listen() 
            print("hi")
            alltx.process_submitted_block()
            #process blocks mined transactions
            alltx.process_mined_transactions() 
            #create summary stats for mined block
            blockdata.process_block_data()
            #create summary stats for last 200 blocks 
            blockdata.analyze_last200blocks() 
             # create summary stats for transactions in last 100 blocks
            alltx.analyzetx_last100blocks()
            #stats for transactions submitted ~ 5m ago
            submitted_5mago = RecentlySubmittedTxDf('5mago', 8, 49, 750000, alltx.df, txpool.txpool_block) 
            #stats for transactions submitted ~ 30m ago
            submitted_30mago = RecentlySubmittedTxDf('30mago', 50, 120, 750000, alltx.df, txpool.txpool_block) 
            #stats for tx in txpool
            txpool.make_txpool_block(alltx.process_block, alltx.alltx_df) 
            #make a prediction table by gas price
            predictiontable = PredictionTable(blockdata, alltx, txpool, submitted5mago, submitted_30mago) 
            #make the gas price report
            gaspricereport = GasPriceReport(predictiontable, blockdata, submitted_5mago, submitted_30mago, array5m, array30m) 
            #hold recent avg gp rec
            array5m = gaspricereport.array5m 
            #hold recent safelow gp rec
            array30m = gaspricereport.array30m 
            #updates tx submitted at current block with data from predictiontable, gpreport- this is for storing in mysql for later optional stats models.
            alltx.update_txblock(txpool.txpool_block, blockdata, predictiontable, gaspricereport.gprecs)
            #update block counter 
            alltx.process_block += 1

        except:
            console.error(traceback.format_exc())


    