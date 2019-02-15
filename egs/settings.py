import configparser
import os
import sys

from web3 import Web3, HTTPProvider, WebsocketProvider, IPCProvider
from .output import Output, OutputException

console = Output()

parser_instance = None
settings_loaded = False
hostname = None

def settings_file_loaded():
    global settings_loaded
    return settings_loaded is True

def load_settings(settings_file=None):
    """Load settings from a settings file."""
    global parser_instance
    global settings_loaded

    if settings_file is None:
        settings_file = get_settings_filepath()

    """Get settings from INI configuration file."""
    parser_instance = configparser.ConfigParser()
    parser_instance.read(settings_file)

    # legacy support for [geth] config section, alternatively [parity]
    # the new name is [rpc]
    if 'rpc' not in parser_instance:
        if 'geth' in parser_instance:
            parser_instance['rpc'] = parser_instance['geth']
        elif 'parity' in parser_instance:
            parser_instance['rpc'] = parser_instance['parity']

    settings_loaded = True

def get_setting(section, name):
    """Get a setting."""
    global parser_instance

    if section in parser_instance:
        if name in parser_instance[section]:
            return parser_instance[section][name]
    raise KeyError("Could not find setting %s.%s in configuration." % (section, name))

def get_settings_filepath():
    """Find a valid configuration file.
    Order of priority:
        '/etc/ethgasstation/settings.conf'
        '/etc/ethgasstation.conf'
        '/etc/default/ethgasstation.conf'
        '/opt/ethgasstation/settings.conf'
    """
    default_ini_locations = [
        '/etc/ethgasstation/settings.conf',
        '/etc/ethgasstation.conf',
        '/etc/default/ethgasstation.conf',
        '/opt/ethgasstation/settings.conf'
    ]

    # short circuit on environment variable
    if "SETTINGS_FILE" in os.environ:
        path = os.path.join(os.getcwd(), os.environ['SETTINGS_FILE'])
        if os.path.isfile(path):
            return os.path.abspath(os.environ['SETTINGS_FILE'])
        else:
            raise FileNotFoundError("Can't find env-set settings file at %s" % path)

    for candidate_location in default_ini_locations:
        if os.path.isfile(candidate_location):
            return candidate_location
    raise FileNotFoundError("Cannot find EthGasStation settings file.")

def get_web3_provider():
    """Get Web3 instance. Supports websocket, http, ipc."""
    global hostname

    protocol = get_setting('rpc', 'protocol')

    if hostname is None:
        hostname = get_setting('rpc', 'hostname')

    port = get_setting('rpc', 'port')

    if hostname.find(".r1.") != -1:
        hostname = hostname.replace(".r1.", ".r2.")
    elif hostname.find(".r2.") != -1:
        hostname = hostname.replace(".r2.", ".r1.")

    console.info("get_web3_provider, RPC hostname => " + hostname)
    
    timeout = 30
    
    #if timeout is None:
    #    try:
    #        timeout = int(get_setting('rpc', 'timeout'))
    #    except KeyError:
    #        timeout = 15 # default timeout is 15 seconds

    if protocol == 'ws' or protocol == 'wss':
        provider = WebsocketProvider(
            "%s://%s:%s" % (
                protocol,
                hostname,
                port),
            websocket_kwargs={'timeout':timeout}
        )
        provider.egs_timeout = timeout
        return Web3(provider)
    elif protocol == 'http' or protocol == 'https':
        provider = HTTPProvider(
            "%s://%s:%s" % (
                protocol,
                hostname,
                port),
            request_kwargs={'timeout':timeout}
        )
        provider.egs_timeout = timeout
        return Web3(provider)
    elif protocol == 'ipc':
        provider = IPCProvider(
            hostname,
            timeout=timeout
        )
        provider.egs_timeout = timeout
        return Web3(provider)
    else:
        raise Exception("Can't set web3 provider type %s" % str(protocol))

def get_mysql_connstr():
    """Get a MySQL connection string for SQLAlchemy, or short circuit to
    SQLite for a dev mode."""
    if "USE_SQLITE_DB" in os.environ:
        sqlite_db_path = os.path.join(os.getcwd(), os.environ["USE_SQLITE_DB"])
        connstr = "sqlite:///%s" % (sqlite_db_path)
        return connstr

    connstr = "mysql+mysqlconnector://%s:%s@%s:%s/%s" % (
        get_setting('mysql', 'username'),
        get_setting('mysql', 'password'),
        get_setting('mysql', 'hostname'),
        get_setting('mysql', 'port'),
        get_setting('mysql', 'database')
        )
    return connstr
