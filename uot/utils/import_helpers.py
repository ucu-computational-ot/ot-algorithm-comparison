import importlib


def import_object(path):
    path = path.split('.')
    module, obj = path[:-1], path[-1]
    module = '.'.join(module)

    module = importlib.import_module(module)
    obj = getattr(module, obj)

    return obj


