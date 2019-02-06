"""
ETH Gas Station
Main Event Loop
"""
import sys
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
    console.info("Type ctl-c to quit and save data to mysql")
    console.info('blocks '+ str(len(blockdata.blockdata_df)))
    console.info('txcount '+ str(len(alltx.df)))

    while True:
        try:
            #get the hashes in the current txpool
            txpool.append_current_txp() 
            #add new pending transactions until new block arrives
            alltx.listen() 
            #process pending transactions
            alltx.process_submitted_block()
            #process blocks mined transactions
            alltx.process_mined_transactions() 
            #create summary stats for mined block
            blockdata.process_block_data(alltx.minedblock_tx_df, alltx.block_obj)
            #create summary stats for last 200 blocks 
            blockdata.analyze_last200blocks(alltx.process_block) 
             # create summary stats for transactions in last 100 blocks
            alltx.analyzetx_last100blocks()
            #stats for tx in txpool
            txpool.make_txpool_block(alltx.process_block, alltx.df) 
            #stats for transactions submitted ~ 5m ago
            submitted_5mago = RecentlySubmittedTxDf('5mago', alltx.process_block, 10, 50, 2000000, alltx.df, txpool) 
            #stats for transactions submitted ~ 30m ago
            submitted_30mago = RecentlySubmittedTxDf('30mago', alltx.process_block, 60, 100, 2000000, alltx.df, txpool) 
            #make a prediction table by gas price
            predictiontable = PredictionTable(blockdata, alltx, txpool, submitted_5mago.df, submitted_30mago.df) 
            #make the gas price report
            gaspricereport = GasPriceReport(predictiontable.predictiondf, blockdata, submitted_5mago, submitted_30mago, array5m, array30m, alltx.process_block) 
            #make predicted wait times
            predictiontable.get_predicted_wait(gaspricereport, submitted_30mago.nomine_gp)
            gaspricereport.get_wait(predictiontable.predictiondf)
            #hold recent avg gp rec
            array5m = gaspricereport.array5m 
            #hold recent safelow gp rec
            array30m = gaspricereport.array30m 
            #updates tx submitted at current block with data from predictiontable, gpreport- this is for storing in mysql for later optional stats models.
            alltx.update_txblock(txpool.txpool_block, blockdata, predictiontable, gaspricereport.gprecs, submitted_30mago.nomine_gp) 
        
            #make report if enabled
            if report_option is True:
                if ((alltx.process_block % 1) == 0):
                    console.info("Generating summary reports for web...")
                    report = SummaryReport(alltx, blockdata)
                    console.info("Writing summary reports for web...")
                    report.write_report()

            gaspricereport.write_to_json()
            predictiontable.write_to_json(txpool)

            if ((alltx.process_block % 1) == 0):
                console.info("Pruning dataframes/mysql from getting too large...")
                blockdata.prune(alltx.process_block)
                alltx.prune(txpool)
                txpool.prune(alltx.process_block) 

            #update counter
            alltx.process_block += 1

        except KeyboardInterrupt:
            #write to mysql
            response = input ("save transactions to mysql (y/n)?")
            if (response.lower() == 'y'):
                alltx.write_to_sql(txpool)
                blockdata.write_to_sql()
                sys.exit()
            else:
                sys.exit()

        except Exception:
            console.error(traceback.format_exc())
            alltx.process_block +=1


    
