from operator import attrgetter
from text_recognizer.learner.exceptions import CancelBatchException, CancelEpochException, CancelFitException
from typing import Sequence

class Callback:
    """
    Abstract base class used to build new callbacks.
    
    Subclass this class and override any of the relevant hooks.
    """
    
    order = 0
    
    def before_batch(self, learner):
        pass
    
    def after_batch(self, learner):
        pass
    
    def before_epoch(self, learner):
        pass
    
    def after_epoch(self, learner):
        pass
    
    def before_fit(self, learner):
        pass
    
    def after_fit(self, learner):
        pass
    
class with_callbacks:
    def __init__(self, name):
        self.name = name
        
    def __call__(self, fn):
        # o is supposed to be an instance of Learner
        def _fn(o, *args, **kwargs):
            try:
                o.callback(f"before_{self.name}")
                fn(o, *args, **kwargs)
                o.callback(f"after_{self.name}")
            except globals()[f"Cancel{self.name.title()}Exception"]:
                pass
            
        return _fn
    
def run_callbacks(callbacks: Sequence[Callback], name: str, learner = None) -> None:
    for callback in sorted(callbacks, key = attrgetter("order")):
        method = getattr(callback, name, None)
        if method is not None:
            method(learner)