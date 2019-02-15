"""
Chaintip Detector

Queries third-party services for the tip of the chain. This can be used
to keep ETH Gas Station in sync, and to throw out data if our geth/parity
node is reporting bad data to us.
"""

import json

import requests
import web3

from .settings import settings_file_loaded, get_setting, get_web3_provider
from .output import Output

class ChaintipException(Exception):
    """Generic exception for Chaintip."""
    pass

class Chaintip(object):
    """Chaintip detection class."""

    get_highest_from = []
    user_agent = "ETHGasStation-ChainTip"
    infura_api_key = None
    etherscan_api_key = None
    enabled = False

    def __init__(self):
        if settings_file_loaded() is False:
            raise ChaintipException("JSONExporter can't get settings.")

        self.web3 = get_web3_provider()
        self.console = Output()

        try:
            sync_tip_with = get_setting('rpc', 'sync_tip_with')
        except KeyError:
            sync_tip_with = False

        if sync_tip_with:
            self.get_highest_from = sync_tip_with.split(',')
            self.enabled = True

        try:
            self.infura_api_key = get_setting('api', 'infura_api_key')
        except KeyError:
            if 'infura' in self.get_highest_from:
                raise ChaintipException("infura_api_key is not set and is required.")

        try:
            self.etherscan_api_key = get_setting('api', 'etherscan_api_key')
        except KeyError:
            if 'etherscan' in self.get_highest_from:
                raise ChaintipException("etherscan_api_key is not set and is required.")



    def get_canonical_highest_block(self):
        """Return an integer value of the highest known block to Chaintip."""
        if self.enabled:
            best, provider = self.get_highest_known_block()
            if best == 0:
                return -1
        else:
            # just get this from our geth/parity
            try:
                #reinitialize provider
                self.web3 = get_web3_provider()
                return self.web3.eth.blockNumber
            except:
                return -1


    def get_highest_known_block(self):
        """Get the highest known block from data sources."""
        highest = []
        providers = []
        for source in self.get_highest_from:
            try:
                if source == 'etherscan':
                    res = self._get_highest_known_block_etherscan()
                    if res is not False:
                        highest.append(res)
                        providers.append('etherscan')
                elif source == 'etherchain':
                    res = self._get_highest_known_block_etherchain()
                    if res is not False:
                        highest.append(res)
                        providers.append('etherchain')
                elif source == 'infura':
                    res = self._get_highest_known_block_infura()
                    if res is not False:
                        highest.append(res)
                        providers.append('infura')
                else:
                    self.console.warn("Unknown blockchain provider %s" % source)
            except:
                self.console.error("Error getting highest block from source %s" % source)
        if len(highest) == 0:
            self.console.error("Do not have a highest block from sources.")
            return (0, 'failure')
        best = max(highest)
        provider = providers[highest.index(best)]
        return (best, provider)


    def _get_highest_known_block_etherscan(self):
        """Get the highest block from etherscan"""
        uri = "https://api.etherscan.io/api"
        res = requests.get(uri,
            data={ 'module': 'proxy', 'action':
                'eth_blockNumber',
                'apikey': self.etherscan_api_key },
            headers= { 'user-agent': self.user_agent },
            timeout=5)

        if res.status_code == 200 and res.json():
            try:
                block_number = int(res.json()['result'], 16)
                return block_number
            except KeyError:
                return False
        return False


    def _get_highest_known_block_etherchain(self):
        """Get the highest block from etherchain.org as nicely as possible"""
        uri = 'https://www.etherchain.org/blocks/data?draw=0&start=0&length=0'
        res = requests.get(uri,
            headers = { 'user-agent': self.user_agent },
            timeout=5)
        if res.status_code == 200:
            try:
                return res.json()['recordsTotal']
            except KeyError:
                return False
        return False


    def _get_highest_known_block_infura(self):
        """Get the highest known block from Consensys Infura"""
        infura_uri = "https://mainnet.infura.io/%s" % (self.infura_api_key)
        try:
            infura_web3 = web3.Web3(web3.HTTPProvider(infura_uri))
            number = infura_web3.eth.blockNumber
            return int(number)
        except:
            self.console.error("Could not retrieve from Infura.")
        return False
