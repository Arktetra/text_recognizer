from copy import copy, deepcopy
from text_recognizer.callbacks import Callback
from text_recognizer.utils import to_cpu
from torcheval.metrics import Mean

class MetricsCallback(Callback):
    def __init__(self, *ms, **metrics):
        for o in ms:
            metrics[type(o).__name__] = o
            
        self.metrics = metrics
        
        self.train_metrics = {}
        self.val_metrics = {}
        for key, value in self.metrics.items():
            self.train_metrics[f"train_{key}"] = value
            self.val_metrics[f"val_{key}"] = deepcopy(value)
                
        self.all_metrics = copy(self.train_metrics)
        self.all_metrics.update(copy(self.val_metrics))
        self.all_metrics["train_loss"] = self.train_loss = Mean()
        self.all_metrics["val_loss"] = self.val_loss = Mean()
        
    def _log(self, log_dict):
        print(log_dict)
        
    def before_fit(self, learner):
        learner.metrics = self
        
    def before_epoch(self, learner):
        [o.reset() for o in self.all_metrics.values()]
        
    def after_epoch(self, learner):
        log = {}
        log["epoch"] = learner.epoch
        log.update({k: f"{v.compute():.4f}" for k, v in self.all_metrics.items()})
        self._log(log)
        
    def after_batch(self, learner):
        x, y = to_cpu(learner.batch)
        
        if learner.training:
            for m in self.train_metrics.values():
                m.update(to_cpu(learner.preds), y)
                
            self.train_loss.update(to_cpu(learner.loss), weight = len(x))
        else:
            for m in self.val_metrics.values():
                m.update(to_cpu(learner.preds), y)
            self.val_loss.update(to_cpu(learner.loss), weight = len(x))