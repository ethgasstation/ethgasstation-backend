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
from sqlalchemy import create_engine, Column, Integer, String, DECIMAL, BigInteger
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Mined_Sql(Base):
    """mysql schema for minedtransaction"""
    __tablename__ = 'minedtx2'
    id = Column(Integer, primary_key=True)
    index = Column(String(75))
    block_mined = Column(Integer)
    block_posted = Column(Integer)
    expectedTime = Column(DECIMAL(5, 2))
    expectedWait = Column(DECIMAL(5, 2))
    mined_probability = Column(DECIMAL(5, 3))
    highgas2 = Column(Integer)
    from_address = Column(String(60))
    gas_offered = Column(Integer)
    gas_price = Column(BigInteger)
    gp10th = Column(DECIMAL(5,2))
    s5mago = Column(Integer)
    hashpower_accepting = Column(Integer)
    hashpower_accepting2 = Column(Integer)
    hgXhpa = Column(Integer)
    miner = Column(String(60))
    num_from = Column(Integer)
    num_to = Column(Integer)
    ico = Column(Integer)
    dump = Column(Integer)
    high_gas_offered = Column(Integer)
    pct_limit = Column(DECIMAL(5, 4))
    removed_block = Column(Integer)
    round_gp_10gwei = Column(Integer)
    time_posted = Column(Integer)
    time_mined = Column(Integer)
    to_address = Column(String(60))
    tx_atabove = Column(Integer)
    tx_unchained = Column(Integer)
    wait_blocks = Column(Integer)
    chained = Column(Integer)
    nonce = Column(Integer)
    average_calc = Column(Integer)
    average_txpool = Column(Integer)
    safelow_calc = Column(Integer)
    safelow_txpool = Column(Integer)

class Tx_Sql(Base):
    """mysql schema for posted transaction"""
    __tablename__ = 'postedtx2'
    id = Column(Integer, primary_key=True)
    index = Column(String(75))
    block_mined = Column(Integer)
    block_posted = Column(Integer)
    expectedTime = Column(DECIMAL(5, 2))
    expectedWait = Column(DECIMAL(5, 2))
    mined_probability = Column(DECIMAL(5, 3))
    from_address = Column(String(60))
    gas_offered = Column(Integer)
    gas_price = Column(BigInteger)
    gp10th = Column(DECIMAL(5,2))
    s5mago = Column(Integer)
    highgas2 = Column(Integer)
    hashpower_accepting = Column(Integer)
    hashpower_accepting2 = Column(Integer)
    hgXhpa = Column(Integer)
    miner = Column(String(60))
    num_from = Column(Integer)
    num_to = Column(Integer)
    ico = Column(Integer)
    dump = Column(Integer)
    high_gas_offered = Column(Integer)
    pct_limit = Column(DECIMAL(5, 4))
    removed_block = Column(Integer)
    round_gp_10gwei = Column(Integer)
    time_posted = Column(Integer)
    time_mined = Column(Integer)
    to_address = Column(String(60))
    tx_atabove = Column(Integer)
    tx_unchained = Column(Integer)
    wait_blocks = Column(Integer)
    nonce = Column(Integer)
    chained = Column(Integer)
    average_calc = Column(Integer)
    average_txpool = Column(Integer)
    safelow_calc = Column(Integer)
    safelow_txpool = Column(Integer)

class Block_Data(Base):
    """mysql schema for block database"""
    __tablename__ = 'blockdata2'
    id = Column(Integer, primary_key=True)
    blockhash = Column(String(75))
    includedblock = Column(Integer)
    mingasprice = Column(Integer)
    blockfee = Column(DECIMAL(25, 5))
    gaslimit = Column(Integer)
    gasused = Column(Integer)
    time_mined = Column(Integer)
    uncsreported = Column(Integer)
    speed = Column(DECIMAL(4, 3))
    miner = Column(String(60))
    numtx = Column(Integer)
    uncle = Column(Integer)
    main = Column(Integer)
    block_number = Column(Integer)

class Timers():
    """
    class to keep track of time relative to network block
    also tracks low mined price from reports
    """
    def __init__(self, start_block):
        self.start_block = start_block
        self.current_block = start_block
        self.process_block = start_block
        self.minlow = 10 #1 gwei
        self.gp_avg_store = np.array([])
        self.gp_safelow_store = np.array([])

    def update_time(self, block):
        self.current_block = block
        self.process_block = self.process_block + 1

    def check_reportblock(self, block):
        if (block - (self.start_block-1))%50 == 0:
            print (str(block) + ' ' + str(self.start_block))
            return True
        return False
    
    def add_block(self, block_number, block_time):
        self.block_store[block_number] = block_time
    
    def read_block_time(self, block_number):
        return self.block_store.pop(block_number, None)

class CleanTx():
    """transaction object / methods for pandas"""
    def __init__(self, tx_obj, block_posted=None, time_posted=None, miner=None):
        self.hash = tx_obj.hash
        self.block_posted = block_posted
        self.block_mined = tx_obj.blockNumber
        self.to_address = tx_obj['to']
        self.from_address = tx_obj['from']
        self.time_posted = time_posted
        self.gas_price = tx_obj['gasPrice']
        self.gas_offered = tx_obj['gas']
        self.round_gp_10gwei()
        self.miner = miner
        self.nonce = tx_obj['nonce']

    def to_dataframe(self):
        data = {self.hash: {'block_posted':self.block_posted, 'block_mined':self.block_mined, 'to_address':self.to_address, 'from_address':self.from_address, 'nonce':self.nonce, 'time_posted':self.time_posted, 'time_mined': None, 'gas_price':self.gas_price, 'miner':self.miner, 'gas_offered':self.gas_offered, 'round_gp_10gwei':self.gp_10gwei}}
        return pd.DataFrame.from_dict(data, orient='index')

    def round_gp_10gwei(self):
        """Rounds the gas price to gwei"""
        gp = self.gas_price/1e8
        if gp >= 1 and gp < 10:
            gp = np.ceil(gp)
        elif gp >= 10:
            gp = gp/10
            gp = np.ceil(gp)
            gp = gp*10
        else:
            gp = 0
        self.gp_10gwei = gp

class CleanBlock():
    """block object/methods for pandas"""
    def __init__(self, block_obj, main, uncle, timemined, mingasprice=None, numtx = None, weightedgp=None, includedblock=None):
        self.block_number = block_obj.number 
        self.gasused = block_obj.gasUsed
        self.miner = block_obj.miner
        self.time_mined = timemined
        self.gaslimit = block_obj.gasLimit 
        self.numtx = numtx
        self.blockhash = block_obj.hash
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



