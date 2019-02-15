# ethgasstation
#### an adaptive gas price oracle for the ethereum blockchain

This is the backend for [ethgasstation](https://ethgasstation.info), written in
Python 3. This python script is designed to monitor a local Geth or Parity node. It will
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

ethgasstation requires **Python 3**, **MySQL/MariaDB**, and **Geth**/**Parity**. You will
need to modify `settings.conf` for your specific environment; some (insecure)
defaults are set to get you up and running.

The oracle outputs JSON files. These files are stored in the output
directory specified by the `settings.conf` file. You may output these JSON
strings to files by setting `json.output_type` to `file` and
`json.output_location` to a filepath, such as:

```
[json]
    output_type = file
    output_location = ./json
```

or you may set `json.output_type` to Redis and give a redis connection string:

```
[json]
    output_type = redis
    output_location = http://localhost:6379
```

Redis password authentication is also supported by adding it to the output
location string, e.g. `http://:password@localhost:6379/`.

### Usage

To run the script as is on bare metal or a VM, manually:

0. Edit `settings.conf` and install to [an allowed directory](https://github.com/ethgasstation/ethgasstation-backend/pull/17/files#diff-bbda44d05044576b25a2c6cf4b0c3597R37).
1. Install requirements using `pip3 install -r requirements.txt`
2. Run `./ethgasstation.py` or `python3 ethgasstation.py`.

If you are running a frontend to ETH Gas Station, use the `--generate-report`
flag to generate detailed JSON reports for front-end or API consumption.

It is also possible to run the oracle as a Docker container.

1. Change the settings in settings.docker.conf.
2. Run `docker build -t ethgasstation-backend .` from this directory.
3. Run `docker run ethgasstation-backend:latest`.

In the Docker service, the Python script will dump data to JSON on Redis.
You will need to update your settings.conf to the internal hostnames
available for MariaDB, Redis, and geth or parity, respectively within your
infrastructure. 


### Deployment

Ensure latest urllib3 is installed

pip install git+https://github.com/shazow/urllib3


or if that doesn't help upgrade requirements.txt, then

cd /usr/local/SettleFinance/ethgasstation-backend
pip install -r requirements.txt

pip install --upgrade urllib3









