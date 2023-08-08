def load_variable(var_name, allowed_variables):
    name_to_var = dict((v.__name__, v) for v in allowed_variables)
    if var_name in name_to_var:
        var = name_to_var[var_name]
    else:
        raise NotImplementedError("Unknown variable: {}".format(var_name))
    return var


def parse_unknown_args(unknown_args) -> dict:
    """
    Turns list of unknown argument segments (from argparse) into a dictionary.
    Arguments are only allowed in these formats:
    -example-arg value
    --example-arg value
    All "-" will be replaced with "_"
    """
    key_val_pairs = zip(*(iter(unknown_args),) *2)    
    dict_args = {}
    for k, v in key_val_pairs:
        if k.startswith("--"):
            k = k.lstrip("--")
        elif k.startswith("-"):
            k = k.lstrip("-")
        else:
            raise ValueError("Additional args should be specified with '-' or '--'")
        k = k.replace("-", "_")
        dict_args[k] = v
    return dict_args
