import json
import os

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        self._set_attributes(config_data)

    def _set_attributes(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config.from_dict(value))
            elif isinstance(value, list):
                setattr(self, key, [Config.from_dict(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    @classmethod
    def from_dict(cls, data):
        config = cls.__new__(cls)
        config._set_attributes(data)
        return config

    def __str__(self):
        return '\n'.join([f"{key}: {value}" for key, value in self.__dict__.items()])
    

def config_to_dict(obj):
    if isinstance(obj, Config):
        return {k: config_to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: config_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [config_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(config_to_dict(v) for v in obj)
    return obj


class ModelConfig(Config):
    def __init__(self, main_config_file, **kwargs):
        with open(main_config_file, 'r') as f:
            config_data = json.load(f)
        for name, obj in kwargs.items():
            if isinstance(obj, str) and os.path.exists(obj):
                with open(obj, 'r') as f:
                    config_data[name] = json.load(f)
            else:
                config_data[name] = obj
        self._set_attributes(config_data)