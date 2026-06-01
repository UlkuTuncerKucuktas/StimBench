MODEL_REGISTRY = {}
EVAL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def register_evaluator(name):
    def decorator(fn):
        EVAL_REGISTRY[name] = fn
        return fn
    return decorator
