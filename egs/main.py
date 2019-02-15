"""
ETH Gas Station
Main Event Loop
"""
import sys
import traceback
import logging
from .egs_ref import *
from .output import Output, OutputException
import datetime
import time
import multiprocessing

console = Output()

def master_control(args):
    console.info("ETH Gas Station, Settle Finance Mod v0.3")
    report_option = False
    if args.generate_report is True:
        report_option = True

    blockdata = BlockDataContainer()
    alltx = AllTxContainer()
    txpool = TxpoolContainer()
    outputMng = OutputManager()
    array5m = []
    array30m = []
    console.info("Type ctl-c to quit and save data to mysql")
    console.info('blocks '+ str(len(blockdata.blockdata_df)))
    console.info('txcount '+ str(len(alltx.df)))
    start_time = time.time()
    op_time = time.time()
    end_time = 0

    def mysqlSave():
        try:
            console.info("Saving 'alltx' sate to MySQL...")
            alltx.write_to_sql(txpool)
            console.info("Saving 'blockdata' sate to MySQL...")
            blockdata.write_to_sql()
        except:
            console.info("FAILED Saving data to MySQL...")

    pMysqlSave = multiprocessing.Process(target = mysqlSave)

    while True:
        try:
            console.info("Started new run at: " + time.strftime("%Y/%m/%d %H:%M:%S") + ", elapsed: " + str(time.time() - start_time) + "s")
            start_time = time.time()

            alltx.reInitWeb3()

            op_time = time.time()
            txpool.append_current_txp()
            console.info("*** get the hashes in the current txpool [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            alltx.listen() 
            console.info("*** add new pending transactions until new block arrives [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            alltx.process_submitted_block()
            console.info("*** process pending transactions [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            alltx.process_mined_transactions() 
            console.info("*** process blocks mined transactions [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            blockdata.process_block_data(alltx.minedblock_tx_df, alltx.block_obj)
            console.info("*** create summary stats for mined block [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            blockdata.analyze_last200blocks(alltx.process_block) 
            console.info("*** create summary stats for last 200 blocks [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            alltx.analyzetx_last100blocks()
            console.info("*** create summary stats for transactions in last 100 blocks [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            txpool.make_txpool_block(alltx.process_block, alltx.df)
            console.info("*** stats for tx in txpool [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            submitted_5mago = RecentlySubmittedTxDf('5mago', alltx.process_block, 10, 50, 2000000, alltx.df, txpool)
            console.info("*** stats for transactions submitted ~ 5m ago [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            submitted_30mago = RecentlySubmittedTxDf('30mago', alltx.process_block, 60, 100, 2000000, alltx.df, txpool) 
            console.info("*** stats for transactions submitted ~ 30m ago [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            predictiontable = PredictionTable(blockdata, alltx, txpool, submitted_5mago.df, submitted_30mago.df)
            console.info("*** make a prediction table by gas price [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            gaspricereport = GasPriceReport(predictiontable.predictiondf, blockdata, submitted_5mago, submitted_30mago, array5m, array30m, alltx.process_block) 
            console.info("*** make the gas price report [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            predictiontable.get_predicted_wait(gaspricereport, submitted_30mago.nomine_gp)
            gaspricereport.get_wait(predictiontable.predictiondf)
            console.info("*** make predicted wait times [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            array5m = gaspricereport.array5m 
            console.info("*** hold recent avg gp rec [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            array30m = gaspricereport.array30m 
            console.info("*** hold recent safelow gp rec [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            alltx.update_txblock(txpool.txpool_block, blockdata, predictiontable, gaspricereport.gprecs, submitted_30mago.nomine_gp) 
            console.info("*** updates tx submitted at current block with data from predictiontable, gpreport- this is for storing in mysql for later optional stats models [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            try:
                console.info("Generating summary reports for web...")
                report = SummaryReport(alltx, blockdata)
                console.info("Writing summary reports for web...")
                report.write_report()
            except Exception as e:
                logging.exception(e)
                console.info("Report Summary Generation failed, see above error ^^")
            console.info("*** always make json report [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            gaspricereport.write_to_json()
            predictiontable.write_to_json(txpool)
            console.info("*** make json for frontend [" + str(time.time() - op_time) + "] s")

            op_time = time.time()
            blockdata.prune(alltx.process_block)
            alltx.prune(txpool)
            txpool.prune(alltx.process_block)
            console.info("*** Pruning dataframes/mysql from getting too large [" + str(time.time() - op_time) + "] s")

            #minute
            if not pMysqlSave.is_alive():
                outputMng.handleGacefullHalt()
                pMysqlSave = multiprocessing.Process(target = mysqlSave)
                pMysqlSave.start()

            #update counter
            alltx.process_block += 1

        except KeyboardInterrupt:
            console.info("KeyboardInterrupt => exit...")
            sys.exit()

        except Exception:
            console.error(traceback.format_exc())
            alltx.process_block +=1


    
