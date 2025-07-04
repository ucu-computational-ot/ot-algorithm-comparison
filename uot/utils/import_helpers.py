import importlib

from uot.utils.exceptions import InvalidConfigurationException


def import_object(path):
    path = path.split('.')
    module, obj = path[:-1], path[-1]
    module = '.'.join(module)

    try:
        module = importlib.import_module(module)
        obj = getattr(module, obj)
    except (AttributeError, ModuleNotFoundError) as ex:
        raise InvalidConfigurationException(str(ex))

    return obj


