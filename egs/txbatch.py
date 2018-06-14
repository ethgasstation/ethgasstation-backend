import requests
import json
from web3 import Web3, HTTPProvider
from eth_utils import is_dict
from web3.utils.datastructures import AttributeDict
from web3.utils.toolz import assoc
from .output import Output, OutputException
from hexbytes import HexBytes

console = Output()

class TxBatch(object):
    """An extremely simple JSON-RPC batch request for web3.eth.getTransaction, to use until web3 adds batch."""

    req = None
    web3 = None

    def __init__(self, web3_provider=None):
        """Initialise a TxBatch."""
        if web3_provider is None:
            self.web3 = egs.settings.get_web3_provider()
        else:
            self.web3 = web3_provider
        self._setRequestFromProvider(self.web3)
    

    def batchRequest(self, method, hex_list):
        """submit and process batchRequest."""
        if len(hex_list) == 0:
            # there's no reason to go make a request for an
            # empty batch.
            return {}

        req_list = []
        idx = 0
        for tx_hash in hex_list:
            req = {
                'json-rpc': '2.0',
                'method': method,
                'params': [ tx_hash ],
                'id': idx
            }
            req_list.append(req)
            idx += 1
        results = self._postBatch(req_list)
        if results is False:
            console.warn("Transaction batch request failed")
            return {}
        
        req_results = {}
        for result in results:
            key = hex_list[int(result['id'])]
            value = self._castAttributeDict(
                        self._formatTransactionResult(
                            result['result']))
            req_results[key] = value

        return req_results

    def _postBatch(self, post_data_object):
        """Make a batch JSON-RPC request to the geth endpoint."""
        try:
            res = requests.post(self.endpoint_uri, json=post_data_object, timeout=5)
            if res.status_code == 200:
                return res.json()
            else:
                return False
        except Exception as e:
            console.warn (e)
            console.warn ("post_batch failure")
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
            print(result)
            for key, value in result.items():
                if isinstance(value, str):
                    strlen = len(value)
                    if strlen == 66:
                        result[key] = HexBytes(value)
                    elif strlen >= 3 and value[0:2] == '0x':
                        result[key] = int(value, 16)
        print(result)
        quit()
        return result

class TxBatchError(Exception):
    pass