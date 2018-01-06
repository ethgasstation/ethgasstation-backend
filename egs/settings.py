import configparser
import os

parser_instance = None
settings_loaded = False

def settings_file_loaded():
    global settings_loaded
    return settings_loaded is True

def load_settings(settings_file):
    """Load settings from a settings file."""
    global parser_instance
    global settings_loaded

    """Get settings from INI configuration file."""
    parser_instance = configparser.ConfigParser()
    parser_instance.read(settings_file)
    settings_loaded = True

def get_setting(section, name):
    """Get a setting."""
    global parser_instance

    if section in parser_instance:
        if name in parser_instance[section]:
            print(parser_instance[section][name])
            return parser_instance[section][name]
    raise KeyError("Could not find setting %s.%s in configuration." % (section, name))

def get_settings_filepath(curdir):
    """Find a valid configuration file.
    Order of priority:
        1. ./settings.conf
        2. /etc/ethgasstation/settings.conf
        3. /etc/ethgasstation.conf"""
    ap = os.path.abspath(curdir)
    if os.path.isfile(os.path.join(ap, 'settings.conf')):
        return os.path.join(ap, 'settings.conf')
    elif os.path.isdir('/etc'):
        if os.path.isdir('/etc/ethgasstation') and \
            os.path.isfile('/etc/ethgasstation/settings.conf'):
            return '/etc/ethgasstation/settings.conf'
        elif os.path.isfile('/etc/ethgasstation.conf'):
            return '/etc/ethgasstation.conf'
    raise FileNotFoundError("Cannot find EthGasStation settings file.")
