import torch
import torch.nn as nn
import pandas as pd
from callbacks import EarlyStop
from MushroomClassifier import MushroomClassifier
from torch.utils.data import DataLoader
import tqdm

class Trainer:
    BATCH_SIZE = 64
    def __init__(self, model: MushroomClassifier, 
                 train_dl: DataLoader, val_dl: DataLoader) -> None:
        self.meta_train = self.init_df()
        self.early_stop = EarlyStop(patience=3)
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.sparcities = []
        self.configure_loss()
        self.configure_optimizer()


    def train_step(self, Xbatch, ybatch):
        self.optimizer.zero_grad()
        y_pred = self.model(Xbatch)
        loss = self.loss_fn(y_pred, ybatch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, X_val, y_val):
        with torch.no_grad():
            y_pred_test = self.model(X_val)
            loss = self.loss_fn(y_pred_test, y_val)
            accuracy_test = (y_pred_test.round() == y_val).float().mean()
        return accuracy_test.item(), loss.item()

    def train(self, epochs:int):
        for epoch in range(1, epochs+1):
            with tqdm.tqdm(self.train_dl, unit="batch") as tepoch:
                for Xbatch, ybatch in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    train_loss = self.train_step(Xbatch.cuda(), ybatch.cuda())
                    tepoch.set_postfix(loss=train_loss)
            val_loss = 0.
            val_acc = 0.
            for _, (X_val_batch, y_val_batch) in enumerate(self.val_dl):
                acc_test, test_loss =self.val_step(X_val_batch.cuda(),
                                        y_val_batch.cuda())
                val_loss += test_loss
                val_acc += acc_test
            val_loss /= len(self.val_dl)
            val_acc /= len(self.val_dl)
            sparcity = self.model.get_sparcity().item()
            row = [epoch, train_loss, val_loss, val_acc, sparcity]
            self.meta_train.loc[len(self.meta_train), :] = row

    def configure_loss(self):
         self.loss_fn = nn.BCELoss()

    def configure_optimizer(self):
         self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                          lr=1e-4, momentum=0.0)
    
    def init_df(self):
        return pd.DataFrame(columns=["epoch", "loss", "val_loss", "test_acc", 'sparcity'])
         
    def prune_smallest_magnitude(self, percent_prune):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name != 'output':
                # computing the mask
                weight_temp, _ = torch.sort(module.weight.data.abs().flatten())
                treshold_ind = int(len(weight_temp) * percent_prune)
                threshold = weight_temp[treshold_ind]
                weight = module.weight.data.abs()
                mask = torch.gt(weight, threshold).float() # R -> Bool, if x > threshold => true
                
                module.mask[mask==0] = 0
                # reassigning the weights
                self.model.prune_weights(name)

    def find_winning_tickets(self, epochs_pretrain:int, 
                             percent_prune: float, pruning_rounds: int,
                             epochs_posttrain:int):
        percent_prune_round = 0
        # pretrain model for x epochs
        self.train(epochs_pretrain)
        for round in range(1, pruning_rounds+1):
            # prune y% of weights with the smallest magnitude
            percent_prune_round += percent_prune
            self.prune_smallest_magnitude(percent_prune_round)
            # set the rest of the weights to the initialization ('rewind weights')
            self.model.rewind_weights()
            # finetune model for 1 epoch
            self.train(1)
        self.model.rewind_weights()
        self.meta_train = self.init_df()
        self.train(epochs_posttrain)

            



        
