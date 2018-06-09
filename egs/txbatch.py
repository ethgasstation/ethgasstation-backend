import requests
import json
from web3 import Web3, HTTPProvider
from eth_utils import is_dict
from web3.utils.datastructures import AttributeDict
from web3.utils.toolz import assoc

from hexbytes import HexBytes

class TxBatch(object):
    """An extremely simple JSON-RPC batch request for web3.eth.getTransaction, to use until web3 adds batch."""

    req = None
    web3 = None

    endpoint_uri = None
    tx_lock = False

    tx_hashes = []

    def __init__(self, web3_provider=None):
        """Initialise a TxBatch."""
        if web3_provider is None:
            self.web3 = egs.settings.get_web3_provider()
        else:
            self.web3 = web3_provider
        self._setRequestFromProvider(self.web3)

    def addTxHash(self, txHash):
        """Add a transaction hash to the pool of those to retrieve."""
        if isinstance(txHash, HexBytes):
            self.tx_hashes.append(txHash.hex().lower())
        elif isinstance(txHash, str):
            self.tx_hashes.append(txHash.lower())
        else:
            raise TypeError("TxBatch.addTxHash: txHash is not a string or HexBytes")

    def addTxHashes(self, txHashes):
        """Queue a list of hashes for a transaction batch request."""
        for txHash in txHashes:
            self.addTxHash(txHash)

    def getTransactions(self, clear_batch_hashes=True):
        """Get all transactions queued with addTxHash/addTxHashes."""
        if len(self.tx_hashes) == 0:
            # there's no reason to go make a request for an
            # empty batch.
            return {}

        req_list = []
        idx = 0
        for tx_hash in self.tx_hashes:
            req = {
                'json-rpc': '2.0',
                'method': 'eth_getTransactionByHash',
                'params': [ tx_hash ],
                'id': idx
            }
            req_list.append(req)
            idx += 1
        results = self._postBatch(req_list)
        if results is False:
            raise TxBatchError("Transaction batch request failed")
        
        req_results = {}
        for result in results:
            key = self.tx_hashes[int(result['id'])]
            value = self._castAttributeDict(
                        self._formatTransactionResult(
                            result['result']))
            req_results[key] = value

        if clear_batch_hashes is True:
            self.tx_hashes.clear()

        return req_results

    def _postBatch(self, post_data_object):
        """Make a batch JSON-RPC request to the geth endpoint."""
        res = requests.post(self.endpoint_uri, json=post_data_object, timeout=5)
        if res.status_code == 200:
            return res.json()
        else:
            return False
    
    def _setRequestFromProvider(self, web3_provider):
        """Get the Geth HTTP endpoint URI from an instantiated Web3 provider."""
        for provider in web3_provider.providers:
            if isinstance(provider, HTTPProvider):
                self.endpoint_uri = provider.endpoint_uri

    def _castAttributeDict(self, maybe_dict):
        """Return an AttributeDict as is provided by web3 middleware."""
        if is_dict(maybe_dict) and not isinstance(maybe_dict, AttributeDict):
            return AttributeDict.recursive(maybe_dict)
        else:
            return maybe_dict

    def _formatTransactionResult(self, result):
        """Get proper types from returned hex."""
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, str):
                    strlen = len(value)
                    if strlen == 66:
                        result[key] = HexBytes(value)
                    elif strlen >= 3 and value[0:2] == '0x':
                        result[key] = int(value, 16)
        return result

class TxBatchError(Exception):
    pass