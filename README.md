# ethgasstation
#### an adaptive gas price oracle for the ethereum blockchain

This is the backend for [ethgasstation](https://ethgasstation.info), written in
Python 3. This python script is designed to monitor a local Geth node. It will
record data about pending and mined transactions, including the transactions in
your node's transaction pool. Its main purpose is to generate adaptive gas price
estimates that enable you to know what gas price to use depending on your
confirmation time needs. It generates these estimates based on miner policy
estimates as well as the number of transactions in the txpool and the gas
offered by the transaction.

The basic strategy is to use statistical modelling to predict confirmation times
at all gas prices from 0-100 gwei at the current state of the txpool and minimum
gas prices accepted in blocks over the last 200 blocks.  Then, it selects the
gas price that gives the desired confirmation time assuming standard gas offered
(higher than 1m gas is slower).

### Installation and Prerequisites

ethgasstation requires **Python 3**, **MySQL/MariaDB**, and **Geth**. You will
need to modify `settings.conf` for your specific environment; some (insecure)
defaults are set to get you up and running.

The oracle outputs JSON files. These files are stored in the output
directory specified by the `settings.conf` file. In the future the Oracle
will support saving JSON blobs to cloud object storage, Redis, MySQL, etc.


### Usage

1. Install requirements using `pip3 install -r requirements.txt`
2. Run `./ethgasstation.py` or `python3 ethgasstation.py`.
