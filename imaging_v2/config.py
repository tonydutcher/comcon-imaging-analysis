
DEFAULT_CONFIG_FILE = 'config_default.py'

def load_config(config_file=DEFAULT_CONFIG_FILE):
    config_base = config_file.split('.')[0]
    return __import__(config_base)

config = load_config()
