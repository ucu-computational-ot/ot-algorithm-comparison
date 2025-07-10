import sys
import os
import yaml
import argparse
from typing import Any

from uot.utils.import_helpers import import_object
from uot.problems.store import ProblemStore
from uot.problems.hdf5_store import HDF5ProblemStore
from uot.utils.logging import setup_logger

logger = setup_logger(__name__)


def _resolve_references(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively import any string references for 'generator', 'cost_fn',
    and nested generator configs ending with '_cfg'.
    Convert 'borders' lists or strings to tuples of floats.
    """
    resolved: dict[str, Any] = {}
    for key, val in cfg.items():
        # skip hidden or anchor defaults
        if key.startswith('__'):
            continue

        # import any class or function by defined path
        if key == 'generator' or key == 'class' or key == 'cost_fn':
            resolved[key] = import_object(val)
        # parse borders param for genereators
        elif key == 'borders':
            # allow list/tuple of numbers or string '(a,b)'
            if isinstance(val, str):
                parts = val.strip('()').split(',')
                resolved[key] = tuple(float(p) for p in parts)
            else:
                resolved[key] = tuple(float(p) for p in val)
        # for the case of nested generators in generators (i.e. paired one)
        elif key.endswith('_cfg') and isinstance(val, dict):
            # nested generator config: has 'class' and optional 'params'
            nested = val.copy()
            # import nested class
            nested['class'] = import_object(nested['class'])
            # if params exist, resolve inside them
            if 'params' in nested:
                nested['params'] = _resolve_references(nested['params'])
            resolved[key] = nested
        else:
            # leave other values (e.g. ints, floats, booleans) as-is
            resolved[key] = val
    return resolved


def serialize_problems(
        config_path: str,
        export_dir: str | None,
        export_hdf5: str | None
) -> None:
    # Load YAML config
    with open(config_path, encoding='utf8') as f:
        raw_cfg = yaml.safe_load(f)

    generators = raw_cfg.get('generators', {})

    if export_hdf5:
        _serialize_problems_hdf5(generators, export_hdf5)
    elif export_dir:
        _serialize_problems_pickle(generators, export_dir)


def _serialize_problems_pickle(generators, export_dir):
    for name, gen_cfg in generators.items():
        # skip hidden or anchor defaults
        if name.startswith('_'):
            continue

        # prepare directory and metadata
        store_dir = os.path.join(export_dir, name)
        os.makedirs(store_dir, exist_ok=True)
        meta_path = os.path.join(store_dir, 'meta.yaml')
        with open(meta_path, 'w', encoding='utf8') as meta_file:
            yaml.dump({name: gen_cfg}, meta_file, default_flow_style=False)

        # resolve imports and parsing
        cfg = {'name': name, **gen_cfg}
        cfg = _resolve_references(cfg)

        generator_cls = cfg.pop('generator')
        cache_gt = cfg.pop('cache_gt', False)
        generator = generator_cls(**cfg)

        logger.info(f"Generating problems for '{name}' using {generator_cls.__name__}")

        # generate and serialize
        store = ProblemStore(store_dir)
        for problem in generator.generate():
            # precompute costs (and optionally ground-truth)
            problem.get_costs()
            if cache_gt:
                problem.get_exact_cost()
            store.save(problem)


def _serialize_problems_hdf5(generators, export_path):
    # os.makedirs(export_path, exist_ok=True)
    store = HDF5ProblemStore(export_path)
    for name, gen_cfg in generators.items():
        # skip hidden or anchor defaults
        if name.startswith('_'):
            continue

        # resolve imports and parsing
        cfg = {'name': name, **gen_cfg}
        cfg = _resolve_references(cfg)

        generator_cls = cfg.pop('generator')
        cache_gt = cfg.pop('cache_gt', False)
        generator = generator_cls(**cfg)

        logger.info(f"Generating problems for '{name}' using {generator_cls.__name__}")

        # generate and serialize
        for problem in generator.generate():
            # precompute costs (and optionally ground-truth)
            problem.get_costs()
            if cache_gt:
                problem.get_exact_cost()
            store.save(problem)


def main():
    parser = argparse.ArgumentParser(
        description='Serialize marginal or two-marginal problems from config'
    )
    parser.add_argument(
        '--config', '-c', type=str, required=True,
        help='Path to YAML config defining generators'
    )
    export_group = parser.add_mutually_exclusive_group(required=True)
    export_group.add_argument(
        '--export-dir', '-o', type=str,
        help='Directory under which each dataset folder will be created\
        using pickle to serialize problems.'
    )
    export_group.add_argument(
        '--export-hdf5', '-e', type=str,
        help='Filename to which store the generated problems\
        using the HDF5 schema.'
    )
    args = parser.parse_args()

    try:
        serialize_problems(
            config_path=args.config,
            export_dir=args.export_dir,
            export_hdf5=args.export_hdf5,
        )
    except Exception as e:
        parser.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
