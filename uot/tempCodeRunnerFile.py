    vars_data = inspect.signature(distribution_map[distribution][dim]).parameters
    vars = list(vars_data.keys() - {'x', 'y', 'z'})