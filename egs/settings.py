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
            return parser_instance[section][name]
    raise KeyError("Could not find setting %s.%s in configuration." % (section, name))

def get_settings_filepath(curdir):
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

    for candidate_location in default_ini_locations:
        if os.path.isfile(candidate_location):
            return candidate_location
    raise FileNotFoundError("Cannot find EthGasStation settings file.")
