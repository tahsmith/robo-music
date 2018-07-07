import os
import sys
import yaml

CONFIG_PATH = os.environ.get('CONFIG_PATH', './config.yml')


def parse_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict


config_dict = parse_config(CONFIG_PATH)


def main(argv):
    print(parse_config(argv[1] if len(argv) > 1 else CONFIG_PATH))


if __name__ == '__main__':
    main(sys.argv)
