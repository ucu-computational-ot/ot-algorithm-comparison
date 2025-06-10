import os
import yaml
import argparse

from uot.utils.import_helpers import import_object
from uot.problems.store import ProblemStore

def parse_config(name, generator_config) -> dict:
    """
    Pase config: load specified generator and cost functions, pasrse borders
    """
    generator_config['generator'] = import_object(generator_config['generator'])
    generator_config['cost_fn'] = import_object(generator_config['cost_fn'])
    generator_config['name'] = name
    generator_config['borders'] = tuple(map(int, generator_config['borders'].strip('()').split(',')))

    return generator_config


def serialize_problems(config_path: str, export_dir: str) -> None:
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    for name, generator_config in config['generators'].items():
        store_dir = os.path.join(export_dir, name)
        problem_store = ProblemStore(store_dir)

        # save additional meta file to know how dataset was generated
        meta_path = os.path.join(store_dir, "meta.yaml")
        with open(meta_path, "w", encoding="utf8") as meta_file:
            yaml.dump({name: generator_config}, meta_file, default_flow_style=False)

        generator_config = parse_config(name, generator_config)

        generator_class = generator_config.pop('generator')
        generator = generator_class(**generator_config)

        problems = generator.generate()

        for problem in problems:
            problem.get_costs() # pre-compute costs to save time in experiments
            problem_store.save(problem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read config path")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--export-dir', type=str, required=True, help='Path to the export file')
    args = parser.parse_args()

    serialize_problems(config_path=args.config, export_dir=args.export_dir)

