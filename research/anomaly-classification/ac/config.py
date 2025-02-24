import json
from dataclasses import dataclass
from pprint import pformat
from ac.utils import parse_unknown_args


class ConfigurationError(Exception):
    pass


@dataclass
class ConfigParam:
    name: None
    type: None
    default: None
    min: None
    max: None
    possible_values: None

    def validate_value(self, value):
        if self.type and type(value) != self.type:
            raise ConfigurationError(
                f'incorrect type for param {self.name}: expected type'
                f' {self.type} but instead got type {type(value)}'
            )
        if self.min and value < self.min:
            raise ConfigurationError(
                f'value for param {self.name} must be >= {self.min}'
            )
        if self.max and value < self.max:
            raise ConfigurationError(
                f'value for param {self.name} must be <= {self.max}'
            )
        if self.possible_values and value not in self.possible_values:
            raise ConfigurationError(
                f'param {self.name} must have one of these values:'
                f' {self.possible_values}'
            )


class Config:
    def __init__(self):
        self.key_to_param = {}
        self.key_to_value = {}
    
    def __repr__(self):
        return pformat(self.key_to_value)

    def register_param(
        self,
        name=None,
        type=None,
        default=None,
        min=None,
        max=None,
        possible_values=None
    ):
        param = ConfigParam(
            name, type, default, min, max, possible_values
        )
        self.key_to_param[param.name] = param

    def set_params_from_args(self, args, overwriting_args):
        for key in self.key_to_param:
            if key in args:
                self.set_param(key, args[key])
            else:
                self.set_param(key, self.key_to_param[key].default)
        if overwriting_args:
            for key, value in overwriting_args.items():
                self.set_param(key, value)

    def set_param(self, key, value):
        param = self.key_to_param[key]
        value = param.type(value)
        param.validate_value(value)
        self.key_to_value[key] = value
    
    def __getattr__(self, key):
        return self.key_to_value[key]

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.key_to_value, f)



def load_config(config_cls, config_path, unknown_args):
    with open(config_path) as f:
        config_dict = json.load(f)    
    config = config_cls(config_dict, parse_unknown_args(unknown_args))
    return config