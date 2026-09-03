# torch-backed registration is optional so a partial checkout with only stimbench/synth imports
try:
    from .registry import MODEL_REGISTRY, EVAL_REGISTRY  # noqa: F401
    import stimbench.models  # noqa: F401
    import stimbench.eval  # noqa: F401
except ImportError:
    pass
