from text_recognizer import BaseModule
from text_recognizer.callbacks import Callback, with_callbacks, run_callbacks
from torch.utils.data import DataLoader
from typing import Optional, Sequence

import torch

TRAIN_DATALOADER = DataLoader
VAL_DATALOADER = DataLoader

class Learner:
    def __init__(
        self,
        accelerator: str = None,
        callbacks: Sequence[Callback] = None,
        epochs: int = 5
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.callbacks = callbacks
        self.epochs = epochs
        self.log_dict = {}
        
    def log(self):
        print(self.log_dict)
        
    @with_callbacks("batch")
    def fit_batch(self, model: BaseModule):
        batch = self.batch[0].to(self.accelerator), self.batch[1].to(self.accelerator)
        
        if self.training:
            loss = model.training_step(batch, self.batch_idx)
            model.backward(loss)
            model.optimizer_step(self.optimizer)
            model.optimizer_zero_grad(self.optimizer)
        else:
            loss = model.validation_step(batch, self.batch_idx)
            
        self.preds = model.logits
        
        self.loss = loss / len(batch)
        
    @with_callbacks("epoch")
    def fit_epoch(
        self, 
        model: BaseModule, 
        train_dataloader: TRAIN_DATALOADER, 
        val_dataloader: VAL_DATALOADER, 
    ):
        self.training = True
        for batch_idx, batch in enumerate(train_dataloader):
            self.batch_idx, self.batch = batch_idx, batch
            self.fit_batch(model)
            
        self.training = False
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                self.batch_idx, self.batch = batch_idx, batch
                self.fit_batch(model)
            
    @with_callbacks("fit")    
    def fit(
        self,
        model: BaseModule,
        train_dataloader: Optional[TRAIN_DATALOADER] = None,
        val_dataloader: Optional[VAL_DATALOADER] = None,
    ) -> None:    
        if self.accelerator == "cuda":
            model.to("cuda")
        
        self.optimizer = model.configure_optimizers()
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.fit_epoch(model, train_dataloader, val_dataloader)
                                
    def callback(self, method_name):
        run_callbacks(self.callbacks, method_name, self)