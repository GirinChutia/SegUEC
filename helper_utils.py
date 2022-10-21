import yaml
from yaml.loader import SafeLoader

def load_yaml(yaml_path):
    with open(yaml_path) as f:
        data = yaml.load(f,Loader=SafeLoader)
    return data