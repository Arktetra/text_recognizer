from typing import Any
from typing_extensions import override

import torch
import torch.nn as nn

class BaseModule(nn.Module):
    """
    Create AyeModule.
        
    Examples::
        
        >>> from text_recognizer import BaseModule
        >>> import torch.nn as n
        >>> import torch.nn.functional as F
        >>>
        >>> class MNISTClassifier(BaseModule):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>
        >>>         self.layer_1 = nn.Linear(28 * 28, 128)
        >>>         self.layer_2 = nn.Linear(128, 10)
        >>>         self.criterion = F.nll_loss
        >>>
        >>>     def forward(self, x):
        >>>         x = x.view(x.size(0), -1)
        >>>         x = self.layer_1(x)
        >>>         x = F.relu(x)
        >>>         x = self.layer_2(x)
        >>>         x = F.log_softmax(x, dim = 1)
        >>>         return x
        >>>     
        >>>     def _shared_step(self, batch, batch_idx):
        >>>         x, y = batch
        >>>         self.logits = self(x)
        >>>         return self.criterion(self.logits, y)
        >>>
        >>>     def training_step(self, batch, batch_idx):
        >>>         return self._shared_step(batch, batch_idx)
        >>>
        >>>     def validation_step(self, batch, batch_idx):
        >>>         return self._shared_step(batch, batch_idx)
        >>>
        >>>     def test_step(self, batch, batch_idx):
        >>>         return self._shared_step(batch, batch_idx)
        >>>
        >>>     def configure_optimizers(self):
        >>>         return torch.optim.Adam(params = self.parameters())
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        
    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)
        
    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("`training_step` must be implemented to be used with the Aye Learner.")
    
    def validation_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass
        
    def test_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass 
    
    def predict_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        batch = kwargs.get("batch", args[0])
        return self(batch)
    
    def configure_optimizers(self):
        raise NotImplementedError("`configure_optimizer` must be implemented to be used with the Aye Learner.")
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()
    
    def optimizer_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad()
    
    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        loss.backward(*args, **kwargs)