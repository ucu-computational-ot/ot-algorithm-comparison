import optax

# ----------------------------------------------------------------------
# Helper: build a schedule from a string + numeric parameters
# ----------------------------------------------------------------------
def _make_schedule(sched: str, init_lr: float, **sched_kwargs):
    """
    Supported schedules:
        "constant"          -> constant learning-rate
        "exponential"       -> init_lr * decay_rate ** (step / decay_steps)
        "cosine"            -> cosine decay to final_lr
        "linear_warmup"     -> linear warm-up for `warmup_steps` then constant
    """
    sched = sched.lower()
    if sched == "constant":
        return optax.constant_schedule(init_lr)

    elif sched == "exponential":
        decay_rate   = sched_kwargs.get("decay_rate", 0.96)
        decay_steps  = sched_kwargs.get("decay_steps", 100)
        return optax.exponential_decay(init_lr, decay_steps, decay_rate)

    elif sched == "cosine":
        total_steps  = sched_kwargs.get("total_steps", 1000)
        final_lr     = sched_kwargs.get("final_lr", 1e-6)
        return optax.cosine_decay_schedule(init_lr, total_steps, final_lr / init_lr)

    elif sched == "linear_warmup":
        warmup_steps = sched_kwargs.get("warmup_steps", 50)
        after_lr     = sched_kwargs.get("after_lr", init_lr)
        return optax.join_schedules(
            [optax.linear_schedule(0.0, init_lr, warmup_steps),
             optax.constant_schedule(after_lr)],
            [warmup_steps]
        )
    else:
        raise ValueError(f"Unknown schedule '{sched}'. Choose from constant, exponential, cosine, linear_warmup.")
