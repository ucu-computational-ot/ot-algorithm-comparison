import inspect
from typing import Type, Any, Dict

def instantiate_solver(
    solver_cls: Type, 
    init_kwargs: Dict[str, Any]
) -> Any:
    """
    Instantiate solver_cls by matching init_kwargs to its __init__ signature.
    Raises if any required parameter is missing.
    """
    sig = inspect.signature(solver_cls.__init__)
    # parameters, excluding 'self'
    params = {
        name: p
        for name, p in sig.parameters.items()
        if name != 'self'
    }

    # Split kwargs into those we can pass, and the rest (ignored)
    passed = {k: v for k, v in init_kwargs.items() if k in params}
    missing = [
        name for name, p in params.items()
        if p.default is inspect.Parameter.empty and name not in passed
    ]
    if missing:
        raise TypeError(
            f"{solver_cls.__name__} missing required init args: {missing}"
        )

    return solver_cls(**passed)
