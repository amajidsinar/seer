import torch.nn as nn   
from tqdm import tqdm
import torch
import pdb
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import numpy as np
from seer.utils import elapsed_timer
from collections import defaultdict
from typing import Dict, Callable, List, Sequence, Optional
import re
from pathlib import Path

__all__ = ['Trainer']

class Trainer():
    def __init__(self, 
                 model_name: str, 
                 trainer_configuration: dict, 
                 model: torch.nn.Module, 
                 dataloaders: torch.utils.data.DataLoader, 
                 loss: torch.nn, 
                 metrics: object, 
                 optimizer: torch.optim, 
                 device: torch.device, 
                 logger: object, 
                 lr_scheduler: torch.optim.lr_scheduler = Optional, 
                 resume: str = Optional) -> None:

        self.model_name = model_name
        self.trainer_configuration = trainer_configuration
        self.model = model
        self.dataloaders = dataloaders
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.n_epochs = trainer_configuration["n_epochs"]
        self.checkpoint_path = trainer_configuration["checkpoint_path"]
        self.logger = logger
        self.start_epoch = 0
        self.save_best_metric = trainer_configuration["save_best_metric"]
        self.best_metric = defaultdict(float)
        self.best_metric['loss'] = 999999

        if resume:
            try:
                print(f'Loading checkpoint at {resume}')
                checkpoint = torch.load(resume)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_metric = checkpoint['best_metric']
                print(f"Last training at epoch {checkpoint['epoch']}, resume training at epoch {self.start_epoch}")
                print("Finished loading checkpoint")
            except FileNotFoundError:
                print(f'No checkpoint found at {resume}')

        # dump optimizer hyperparams
        optimizer_name = optimizer.__class__.__name__
        self.logger.log_parameter(name = "optimizer", value = optimizer_name)
        for optimizer_param_name, optimizer_param_value in optimizer.param_groups[0].items():
            if optimizer_param_name == 'params':
                pass
            else:
                logger.log_parameter(name = optimizer_param_name, value = optimizer_param_value)
        
        # dump loss
        loss_name = loss.__class__.__name__
        self.logger.log_parameter(name = "loss", value = loss_name)

        # dump scheduler
        if self.lr_scheduler:
            self.lr_scheduler_name = lr_scheduler.__class__.__name__
            self.logger.log_parameter(name = 'scheduler', value = self.lr_scheduler_name)
            for scheduler_param_name, scheduler_param_value in self.lr_scheduler.__dict__.items():
                logger.log_parameter(name = scheduler_param_name, value = scheduler_param_value)

    def _calculate_and_log_aggregate_metrics(self, dataset, epoch):
        print("###################################################################")
        performance_metrics = defaultdict(list)
        print(f'Logging into comet-ml')
        for metric in self.metrics:
            metric_name = metric.__name__
            metric_value = metric(self.prediction_output['targets'], self.prediction_output['predictions'])
            performance_metrics[metric_name] = metric_value
            self.logger.log_metric(f'{dataset} {metric_name}', metric_value, step = epoch)
            print(f'{metric_name}: {metric_value}')
        
        return performance_metrics

    def _calculate_and_log_running_metrics(self, dataset, epoch):
        performance_metrics = defaultdict(list)
        print(f'Logging into comet-ml')
        for metric_name, metric_values in self.running_metrics.items():
            if metric_name == 'fps':
                metric_value = np.mean(metric_values)
                performance_metrics[metric_name] = np.mean(metric_value)
            elif metric_name == 'loss':
                metric_value = np.sum(metric_values)
            performance_metrics[metric_name] = metric_value
            print(f'{metric_name}: {metric_value}')
            self.logger.log_metric(f'{dataset} {metric_name}', metric_value, step = epoch)
        return performance_metrics


    def _train_for_single_dataset(self, dataset: dict, epoch: int):
        self.running_metrics = defaultdict(list)
        self.prediction_output = defaultdict(list)

        
        if bool(re.search("train", dataset)):
            print(f'Training {dataset}')
            self.model.train()
        else:
            print(f'Validating {dataset}')
            self.model.eval()


        for inputs, targets in tqdm(self.dataloaders[dataset]):   
            with elapsed_timer() as elapsed:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # make sure the gradient is zero before forward pass
                self.optimizer.zero_grad()
                # forward pass
                # track history only in train
                with torch.set_grad_enabled(bool(re.search('train', dataset))):
                    outputs = self.model(inputs)
                    batch_loss = self.loss(outputs, targets)

                    outputs = torch.softmax(outputs, dim=1)
                    confidence_scores, predictions = torch.max(outputs, dim=1)
                    
                    confidence_scores, predictions = confidence_scores.detach().cpu().numpy(), predictions.detach().cpu().numpy()
                    targets = targets.detach().cpu().numpy()
                    
                    if bool(re.search("train", dataset)):
                        batch_loss.backward()
                        self.optimizer.step()
                    
                    self.running_metrics['loss'].append(batch_loss.item() / len(targets))
                    self.running_metrics['fps'].append(len(targets) / elapsed())
                    self.prediction_output['confidence_scores'].extend(confidence_scores)
                    self.prediction_output['predictions'].extend(predictions)
                    self.prediction_output['targets'].extend(targets)
                    
        performance_metrics = self._calculate_and_log_aggregate_metrics(dataset, epoch)
        performance_metrics.update(self._calculate_and_log_running_metrics(dataset, epoch))
        
        return performance_metrics


    def _train_for_all_datasets_in_single_epoch(self, epoch: int):
        training_performance_metrics = defaultdict(list)
        validation_performance_metrics = defaultdict(list)
        lr = self.optimizer.param_groups[0]["lr"]
        print(f'Learning rate: {lr}')
        self.logger.log_metric(name = 'lr', value = lr, step = epoch)
        for dataset in self.dataloaders.keys():
            single_dataset_performance_metrics = self._train_for_single_dataset(dataset, epoch)
            
            for metric_name, metric_value in single_dataset_performance_metrics.items():
                
                if bool(re.search('train', dataset)):
                    training_performance_metrics[metric_name].append(metric_value)
                    
                elif bool(re.search('val', dataset)):
                    validation_performance_metrics[metric_name].append(metric_value)
            
        #pdb.set_trace()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        

        training_performance_metrics = self._calculate_and_log_average_metrics(training_performance_metrics, epoch, "average training")
        print(f'#################################################################################')
        validation_performance_metrics = self._calculate_and_log_average_metrics(validation_performance_metrics, epoch, "average validation")
        print(f'#################################################################################')        

        return validation_performance_metrics

    def _calculate_and_log_average_metrics(self, performance_metrics, epoch, prefix):
        for metric_name, metric_value in performance_metrics.items():
            average_metric_value = np.mean(metric_value)
            print(f'{prefix} {metric_name}: {average_metric_value}')
            self.logger.log_metric(f'{prefix} {metric_name}', average_metric_value, step = epoch)
            performance_metrics[metric_name] = average_metric_value
        return performance_metrics

    
    def _save_model(self, suffix : str, best_metric: dict, epoch: int):

        checkpoint_pth = f'{self.checkpoint_path}/{self.model_name}_{suffix}.pth'
        patience_counter = 0

        print(f'Saving training model at epoch {epoch}')
        print(f'Saving training model at {checkpoint_pth}')

        model_params = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'experiment_key': self.logger.get_key()
            }
        #pdb.set_trace()
        model_params.update({'best_metric': best_metric})
        torch.save(model_params, checkpoint_pth)
        
    def train(self):
        current_metric = defaultdict(float)

        for epoch in tqdm(range(self.start_epoch, self.n_epochs)):
            print(f'Epoch: {epoch}')
            validation_performance_metrics = self._train_for_all_datasets_in_single_epoch(epoch)

            for metric_name, metric_value in validation_performance_metrics.items():
                for best_metric_name in self.save_best_metric:
                    if best_metric_name == metric_name:
                        current_metric[metric_name] = metric_value
                        print(f'Current {metric_name}: {metric_value} Best {metric_name}: {self.best_metric[metric_name]}')
            
            # conditional for saving model based on best metric value
            for metric_name in self.save_best_metric:
                if metric_name == 'loss':
                    if current_metric[metric_name] < self.best_metric[metric_name]:
                        self._save_model(suffix = metric_name, best_metric = current_metric, epoch = epoch)
                        self.best_metric[metric_name] = current_metric[metric_name]
                        self.logger.log_metric(f'Best {metric_name}', current_metric[metric_name], step = epoch)
                else:
                    if current_metric[metric_name] > self.best_metric[metric_name]:
                        self._save_model(suffix = metric_name, best_metric = current_metric, epoch = epoch)
                        self.best_metric[metric_name] = current_metric[metric_name]
                        self.logger.log_metric(f'Best {metric_name}', current_metric[metric_name], step = epoch)

            # save model for each epoch
            self._save_model(suffix = 'last_epoch', best_metric = self.best_metric, epoch = epoch)

def set_parameter_requires_grad(model: torch.nn.Module, feature_extracting: bool):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
