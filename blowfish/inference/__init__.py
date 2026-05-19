from .scorer import AmbiguityScorer

__all__ = ["AmbiguityScorer", "FeedbackDecider"]


def __getattr__(name):
    if name == "FeedbackDecider":
        from .decider import FeedbackDecider
        return FeedbackDecider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
