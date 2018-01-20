import configparser
import os
import sys

from web3 import Web3, HTTPProvider

parser_instance = None
settings_loaded = False

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
    """Get Web3 instance."""
    web3 = Web3(
        HTTPProvider(
            "%s://%s:%s" % (
                get_setting('geth', 'protocol'),
                get_setting('geth', 'hostname'),
                get_setting('geth', 'port'))))
    return web3

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
