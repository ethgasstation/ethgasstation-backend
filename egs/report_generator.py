import pandas as pd
import numpy as np
import json
import urllib
import time


class SummaryReport(object):
    """analyzes data from last x blocks to create summary stats"""
    def __init__(self, tx_df, block_df, end_block):
        self.end_block = end_block
        self.tx_df = tx_df
        self.block_df = block_df
        self.post = {}

        def get_minedgasprice(row):
            """returns gasprice in gwei if mined otherwise nan"""
            if ~np.isnan(row['block_mined']):
                return row['round_gp_10gwei']/10
            else:
                return np.nan

        self.tx_df['minedGasPrice'] = self.tx_df.apply(get_minedgasprice, axis=1)
        self.tx_df['gasCat1'] = (self.tx_df['minedGasPrice'] <= 1) & (self.tx_df['minedGasPrice'] >=0)
        self.tx_df['gasCat2'] = (self.tx_df['minedGasPrice']>1) & (self.tx_df['minedGasPrice']<= 4)
        self.tx_df['gasCat3'] = (self.tx_df['minedGasPrice']>4) & (self.tx_df['minedGasPrice']<= 20)
        self.tx_df['gasCat4'] = (self.tx_df['minedGasPrice']>20) & (self.tx_df['minedGasPrice']<= 50)
        self.tx_df['gasCat5'] = (self.tx_df['minedGasPrice']>50)
        self.block_df['emptyBlocks'] = (self.block_df['numtx']==0).astype(int)
        self.tx_df['mined'] = self.tx_df['block_mined'].notnull()
        self.tx_df['delay'] = self.tx_df['block_mined'] - self.tx_df['block_posted']
        self.tx_df['delay2'] = self.tx_df['time_mined'] - self.tx_df['time_posted']
        self.tx_df.loc[self.tx_df['delay'] <= 0, 'delay'] = np.nan
        self.tx_df.loc[self.tx_df['delay2'] <= 0, 'delay2'] = np.nan
        total_tx = len(self.tx_df.loc[self.tx_df['minedGasPrice'].notnull()])
        self.post['latestblockNum'] = int(self.end_block)
        self.post['totalTx'] = int(total_tx)
        self.post['totalCatTx1'] = int(self.tx_df['gasCat1'].sum())
        self.post['totalCatTx2'] = int(self.tx_df['gasCat2'].sum())
        self.post['totalCatTx3'] = int(self.tx_df['gasCat3'].sum())
        self.post['totalCatTx4'] = int(self.tx_df['gasCat4'].sum())
        self.post['totalCatTx5'] = int(self.tx_df['gasCat5'].sum())
        self.post['totalTransfers'] = len(self.tx_df[self.tx_df['gas_offered']==21000])
        self.post['avgTxFee'] = self.tx_df.loc[self.tx_df['gas_offered']==21000, 'minedGasPrice'].median()
        self.post['totalConCalls'] = len(self.tx_df[self.tx_df['gas_offered']!=21000])
        self.post['maxMinedGasPrice'] = float(self.tx_df['minedGasPrice'].max())
        self.post['minMinedGasPrice'] = float(self.tx_df['gas_price'].min()/1e9)
        self.post['medianGasPrice']= float(self.tx_df['minedGasPrice'].quantile(.5))
        self.post['cheapestTx'] = float(self.tx_df.loc[self.tx_df['gas_offered']==21000, 'minedGasPrice'].min())
        self.post['cheapestTxID'] = (self.tx_df.loc[(self.tx_df['minedGasPrice']==self.post['cheapestTx']) & (self.tx_df['gas_offered'] == 21000)].index[0]).lower()
        self.post['dearestTx'] = float(self.tx_df.loc[self.tx_df['gas_offered']==21000, 'minedGasPrice'].max())
        self.post['dearestTxID'] = (self.tx_df.loc[(self.tx_df['minedGasPrice']==self.post['dearestTx']) & (self.tx_df['gas_offered'] == 21000)].index[0]).lower()
        self.post['dearestgpID'] = (self.tx_df.loc[self.tx_df['minedGasPrice']==self.post['maxMinedGasPrice']].index[0]).lower()
        self.post['emptyBlocks'] =  len(self.block_df[self.block_df['speed']==0])
        self.post['fullBlocks'] = len(self.block_df[self.block_df['speed']>=.95])
        self.post['totalBlocks'] = len(self.block_df)
        self.post['medianDelay'] = float(self.tx_df['delay'].quantile(.5))
        self.post['medianDelayTime'] = float(self.tx_df['delay2'].quantile(.5))

        """ETH price data"""
        # TODO: cache this URI call somewhere
        # TODO: cert pin cryptocompare so these are less likely to be tampered with
        url = "https://min-api.cryptocompare.com/data/price?fsym=ETH&tsyms=USD,EUR,GBP,CNY"
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                pricesRaw = json.loads(response.read().decode())
            ethPricesTable = pd.DataFrame.from_dict(pricesRaw, orient='index')
            self.post['ETHpriceUSD'] = int(ethPricesTable.loc['USD', 0])
            self.post['ETHpriceEUR'] = int(ethPricesTable.loc['EUR', 0])
            self.post['ETHpriceCNY'] = int(ethPricesTable.loc['CNY', 0])
            self.post['ETHpriceGBP'] = int(ethPricesTable.loc['GBP', 0])
        except:
            self.post['ETHpriceUSD'] = 0
            self.post['ETHpriceEUR'] = 0
            self.post['ETHpriceCNY'] = 0
            self.post['ETHpriceGBP'] = 0

        """find minimum gas price with at least 50 transactions mined"""
        tx_grouped_price = self.tx_df[['block_posted', 'minedGasPrice']].groupby('minedGasPrice').count()
        tx_grouped_price.rename(columns = {'block_posted': 'count'}, inplace = True)
        tx_grouped_price['sum'] = tx_grouped_price['count'].cumsum()
        minlow_series = tx_grouped_price[tx_grouped_price['sum']>50].index
        self.post['minLow'] = float(minlow_series.min())
        self.minlow = float(minlow_series.min()*10)

        """generate table with key miner stats"""
        miner_txdata = self.tx_df[['block_posted', 'miner']].groupby('miner').count()
        miner_txdata.rename(columns={'block_posted':'count'}, inplace = True)
        # Next Find Each Miners Mininum Price of All Mined Transactions
        minprice_df = self.tx_df[['miner', 'minedGasPrice']].groupby('miner').min()
        minprice_df = minprice_df.rename(columns={"minedGasPrice": 'minGasPrice'})
        avgprice_df = self.tx_df[['miner', 'minedGasPrice']].groupby('miner').mean()
        avgprice_df = avgprice_df.rename(columns={"minedGasPrice": 'avgGasPrice'})
        miner_txdata = pd.concat([miner_txdata, minprice_df], axis = 1)
        miner_txdata = pd.concat([miner_txdata, avgprice_df], axis = 1)

        # Calculate Each Miners % Empty and Total Blocks
        miner_blocks = block_df[['miner','emptyBlocks','block_number']].groupby('miner').agg({'emptyBlocks':'sum', 'block_number':'count'})
        miner_txdata = pd.concat([miner_txdata, miner_blocks], axis = 1)
        miner_txdata.reset_index(inplace=True)
        miner_txdata = miner_txdata.rename(columns={'index':'miner', 'block_number':'totBlocks'})
        #Convert to percentages
        total_blocks = miner_txdata['totBlocks'].sum()
        miner_txdata['pctTot'] = miner_txdata['totBlocks']/total_blocks*100
        miner_txdata['pctEmp'] = miner_txdata['emptyBlocks']/miner_txdata['totBlocks']*100
        miner_txdata['txBlocks'] = miner_txdata['totBlocks'] - miner_txdata['emptyBlocks']
        tot_txblocks = miner_txdata['txBlocks'].sum()
        miner_txdata['pctTxBlocks'] = miner_txdata['txBlocks']/tot_txblocks*100
        pct_txblocks = tot_txblocks/total_blocks
        miner_txdata  = miner_txdata.sort_values(['minGasPrice','totBlocks'], ascending = [True, False])
        #Make Table with top10 Miner Stats
        top_miners = miner_txdata.sort_values('totBlocks', ascending=False)
        top_miners = top_miners.loc[:,['miner','minGasPrice','avgGasPrice', 'pctTot']].head(10)
        top_miners = top_miners.sort_values(['minGasPrice','avgGasPrice'], ascending = [True, True]).reset_index(drop=True)
        #Table with hashpower by gasprice
        price_table = miner_txdata[['pctTxBlocks', 'minGasPrice']].groupby('minGasPrice').sum().reset_index()
        price_table['pctTotBlocks'] = price_table['pctTxBlocks'] * pct_txblocks
        price_table['cumPctTxBlocks'] = price_table['pctTxBlocks'].cumsum()
        price_table['cumPctTotBlocks'] = price_table['pctTotBlocks'].cumsum()
        #store dataframes for json
        self.miner_txdata = miner_txdata
        self.top_miners = top_miners
        self.price_table = price_table

        """gas guzzler table"""
        gg = {
            '0x6090a6e47849629b7245dfa1ca21d94cd15878ef': 'ENS registrar',
            '0xcd111aa492a9c77a367c36e6d6af8e6f212e0c8e': 'Acronis',
            '0x209c4784ab1e8183cf58ca33cb740efbf3fc18ef': 'Poloniex',
            '0xd91e45416bfbbec6e2d1ae4ac83b788a21acf583': 'Etheroll',
            '0xa74476443119a942de498590fe1f2454d7d4ac0d': 'Golem',
            '0xedce883162179d4ed5eb9bb2e7dccf494d75b3a0': 'Bittrex',
            '0x70faa28a6b8d6829a4b1e649d26ec9a2a39ba413': 'Shapeshift',
            '0xff1f9c77a0f1fd8f48cfeee58b714ca03420ddac': 'e4row',
            '0x8d12a197cb00d4747a1fe03395095ce2a5cc6819': 'Etherdelta',
            '0xe94b04a0fed112f3664e45adb2b8915693dd5ff3': 'Bittrex Safe Split',
            '0xace62f87abe9f4ee9fd6e115d91548df24ca0943': 'Monaco',
            '0xb9e7f8568e08d5659f5d29c4997173d84cdf2607': 'Swarm City',
            '0x06012c8cf97bead5deae237070f9587f8e7a266d': 'Cryptokitties',
            '0xb1690c08e213a35ed9bab7b318de14420fb57d8c': 'Cryptokitties Auction',
            '0x2a0c0dbecc7e4d658f48e01e3fa353f44050c208': 'IDEX'
        }
        gasguzz = self.tx_df.groupby('to_address').count()
        gasguzz = gasguzz.sort_values('block_mined', ascending = False)
        tottx = len(self.tx_df)
        gasguzz['pcttot'] = gasguzz['block_mined']/tottx*100
        gasguzz = gasguzz.head(n=10)
        for index, row in gasguzz.iterrows():
            if index in gg.keys():
                gasguzz.loc[index, 'ID'] = gg[index]
            else:
                gasguzz.loc[index, 'ID'] = ''
        gasguzz = gasguzz.reset_index()
        self.gasguzz = gasguzz

        """low gas price tx watch list"""
        recent = self.end_block - 250
        lowprice = self.tx_df.loc[(self.tx_df['round_gp_10gwei'] < 10) & (self.tx_df['block_posted'] < recent), ['minedGasPrice', 'block_posted', 'mined', 'block_mined', 'round_gp_10gwei']]
        lowprice = lowprice.sort_values(['round_gp_10gwei'], ascending = True).reset_index()
        lowprice['gasprice'] = lowprice['round_gp_10gwei']/10
        grouped_lowprice = lowprice.groupby('gasprice', as_index=False).head(10)
        grouped_lowprice.reset_index(inplace=True)
        self.lowprice = grouped_lowprice.sort_values('gasprice', ascending=False)

        """average block time"""
        blockinterval = self.block_df[['block_number', 'time_mined']].diff()
        blockinterval.loc[blockinterval['block_number'] > 1, 'time_mined'] = np.nan
        blockinterval.loc[blockinterval['block_number']< -1, 'time_mined'] = np.nan
        self.avg_timemined = blockinterval['time_mined'].mean()

        """median wait time by gas price for bar graph"""
        price_wait = self.tx_df.loc[:, ['minedGasPrice', 'delay2']]
        price_wait.loc[price_wait['minedGasPrice']>=40, 'minedGasPrice'] = 40
        price_wait = price_wait.loc[(price_wait['minedGasPrice']<=10) | (price_wait['minedGasPrice']==20) | (price_wait['minedGasPrice']==21) |(price_wait['minedGasPrice'] == 40), ['minedGasPrice', 'delay2']]
        price_wait.loc[price_wait['minedGasPrice']<1, 'minedGasPrice'] = 0
        price_wait = price_wait.groupby('minedGasPrice').median()/60
        price_wait.reset_index(inplace=True)
        self.price_wait = price_wait