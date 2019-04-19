import pandas as pd
import numpy as np
import json
import string
import urllib
import time
from .jsonexporter import JSONExporter, JSONExporterException
from .output import Output, OutputException
import egs.settings
egs.settings.load_settings()
exporter = JSONExporter()
console = Output()

class SummaryReport():
    """analyzes data from last x blocks to create summary stats"""
    def __init__(self, alltx, blockdata):
        self.end_block = alltx.process_block
        self.tx_df = alltx.df[alltx.df['block_mined'] > (alltx.process_block-1500)].copy()
        self.block_df = blockdata.blockdata_df[blockdata.blockdata_df['block_number'] > (alltx.process_block-1500)].copy()
        self.block_time = blockdata.block_time
        self.post = {}

        def get_minedgasprice(row):
            """returns gasprice in gwei if mined otherwise nan"""
            if ~np.isnan(row['block_mined']):
                return row['round_gp_10gwei']/100
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
        self.tx_df['delay2'] = self.tx_df['delay'] * self.block_time
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
        miner_blocks = self.block_df[['miner','emptyBlocks','block_number']].groupby('miner').agg({'emptyBlocks':'sum', 'block_number':'count'})
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
            '0x32Be343B94f860124dC4fEe278FDCBD38C102D88': 'Poloniex 1',
            '0x209c4784ab1e8183cf58ca33cb740efbf3fc18ef': 'Poloniex 2',
            '0xb794F5eA0ba39494cE839613fffBA74279579268': 'Poloniex 3',
            '0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE': 'Binance 1',
            '0xD551234Ae421e3BCBA99A0Da6d736074f22192FF': 'Binance 2',
            '0x564286362092D8e7936f0549571a803B203aAceD': 'Binance 3',
            '0x0681d8Db095565FE8A346fA0277bFfdE9C0eDBBF': 'Binance 4',
            '0xfE9e8709d3215310075d67E3ed32A380CCf451C8': 'Binance 5',
            '0x4E9ce36E442e55EcD9025B9a6E0D88485d628A67': 'Binance 6',
            '0x120A270bbC009644e35F0bB6ab13f95b8199c4ad': 'Shapeshift 1',
            '0x9e6316f44BaEeeE5d41A1070516cc5fA47BAF227': 'Shapeshift 2',
            '0x70faa28A6B8d6829a4b1E649d26eC9a2a39ba413': 'Shapeshift 3',
            '0x563b377A956c80d77A7c613a9343699Ad6123911': 'Shapeshift 4',
            '0xD3273EBa07248020bf98A8B560ec1576a612102F': 'Shapeshift 5',
            '0x3b0BC51Ab9De1e5B7B6E34E5b960285805C41736': 'Shapeshift 6',
            '0xeed16856D551569D134530ee3967Ec79995E2051': 'Shapeshift 7',
            '0xd91e45416bfbbec6e2d1ae4ac83b788a21acf583': 'Etheroll',
            '0xa74476443119a942de498590fe1f2454d7d4ac0d': 'Golem',
            '0xedce883162179d4ed5eb9bb2e7dccf494d75b3a0': 'Bittrex',
            '0xff1f9c77a0f1fd8f48cfeee58b714ca03420ddac': 'e4row',
            '0x8d12a197cb00d4747a1fe03395095ce2a5cc6819': 'Etherdelta',
            '0xe94b04a0fed112f3664e45adb2b8915693dd5ff3': 'Bittrex Safe Split',
            '0xace62f87abe9f4ee9fd6e115d91548df24ca0943': 'Monaco',
            '0xb9e7f8568e08d5659f5d29c4997173d84cdf2607': 'Swarm City',
            '0x06012c8cf97bead5deae237070f9587f8e7a266d': 'Cryptokitties',
            '0xb1690c08e213a35ed9bab7b318de14420fb57d8c': 'Cryptokitties Auction',
            '0x2a0c0dbecc7e4d658f48e01e3fa353f44050c208': 'IDEX',
            '0x0000000000085d4780b73119b644ae5ecd22b376': 'TrueUSD',
            '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2': 'WrappedEther',
            '0xaB5C66752a9e8167967685F1450532fB96d5d24f': 'Huobi 1',
            '0x6748F50f686bfbcA6Fe8ad62b22228b87F31ff2b': 'Huobi 2',
            '0xfdb16996831753d5331fF813c29a93c76834A0AD': 'Huobi 3',
            '0xeEe28d484628d41A82d01e21d12E2E78D69920da': 'Huobi 4',
            '0x5C985E89DDe482eFE97ea9f1950aD149Eb73829B': 'Huobi 5',
            '0xDc76CD25977E0a5Ae17155770273aD58648900D3': 'Huobi 6',
            '0xadB2B42F6bD96F5c65920b9ac88619DcE4166f94': 'Huobi 7',
            '0xa8660c8ffD6D578F657B72c0c811284aef0B735e': 'Huobi 8',
            '0x1062a747393198f70F71ec65A582423Dba7E5Ab3': 'Huobi 9',
            '0xE93381fB4c4F14bDa253907b18faD305D799241a': 'Huobi 10',
            '0xFA4B5Be3f2f84f56703C42eB22142744E95a2c58': 'Huobi 11',
            '0x46705dfff24256421A05D056c29E81Bdc09723B8': 'Huobi 12',
            '0x1B93129F05cc2E840135AAB154223C75097B69bf': 'Huobi 14',
            '0xEB6D43Fe241fb2320b5A3c9BE9CDfD4dd8226451': 'Huobi 15',
            '0x956e0DBEcC0e873d34a5e39B25f364b2CA036730': 'Huobi 16',
            '0x03cb0021808442Ad5EFb61197966aef72a1deF96': 'coToken'
        }
        gasguzz = self.tx_df.groupby('to_address').sum()
        gasguzz = gasguzz.sort_values('gasused', ascending = False)
        totgas = self.tx_df['gasused'].sum()
        gasguzz['pcttot'] = gasguzz['gasused']/totgas*100
        gasguzz = gasguzz[['gasused', 'pcttot']].head(n=100)
        for index, row in gasguzz.iterrows():
            if index in gg.keys():
                gasguzz.loc[index, 'ID'] = gg[index]
            else:
                gasguzz.loc[index, 'ID'] = ''
        gasguzz = gasguzz.reset_index()
        self.gasguzz = gasguzz

        """low gas price tx watch list"""
        recent = self.end_block - 250
        lowprice = self.tx_df.loc[(self.tx_df['round_gp_10gwei'] < 100) & (self.tx_df['block_posted'] < recent), ['minedGasPrice', 'block_posted', 'mined', 'block_mined', 'round_gp_10gwei']]
        lowprice = lowprice.sort_values(['round_gp_10gwei'], ascending = True).reset_index()
        lowprice['gasprice'] = lowprice['round_gp_10gwei']/100
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
        price_wait = price_wait.dropna(axis=0, subset=['minedGasPrice'])
        price_wait['minedGasPrice'] = price_wait['minedGasPrice'].astype(int)
        price_wait = price_wait.groupby('minedGasPrice').median()/60
        price_wait.reset_index(inplace=True)
        self.price_wait = price_wait

    def write_report(self):
        """write json data"""
        global exporter
        top_minersout = self.top_miners.to_json(orient='records')
        minerout = self.miner_txdata.to_json(orient='records')
        gasguzzout = self.gasguzz.to_json(orient='records')
        lowpriceout = self.lowprice.to_json(orient='records')
        price_waitout = self.price_wait.to_json(orient='records')

        try:
            exporter.write_json('txDataLast10k', self.post)
            exporter.write_json('topMiners', top_minersout)
            exporter.write_json('priceWait', price_waitout)
            exporter.write_json('miners', minerout)
            exporter.write_json('gasguzz', gasguzzout)
            exporter.write_json('validated', lowpriceout)
        except Exception as e:
            console.error("write_report: Exception caught: " + str(e))