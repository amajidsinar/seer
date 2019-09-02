import torch.nn as nn   
from tqdm import tqdm
import torch
import pdb
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import numpy as np
from nf_trainer.utils import elapsed_timer
from collections import defaultdict
from typing import Dict, Callable, List, Sequence, Optional
import re
from pathlib import Path


class Trainer ():
    def __init__(self, model_name, trainer_configuration, model, dataloaders, loss, metrics, optimizer, lr_scheduler, device, logger, resume = Optional):
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
        for hyperparam_name, hyperparam_value in optimizer.param_groups[0].items():
            if hyperparam_name == 'params':
                pass
            else:
                logger.log_parameter(name = hyperparam_name, value = hyperparam_value)
        
        # dump 
        loss_name = loss.__class__.__name__
        self.logger.log_parameter(name = "loss", value = loss_name)


    def _train_for_single_dataset(self, dataset, epoch):
        running_metrics = defaultdict(list)
        output = defaultdict(list)
        performance_metrics = defaultdict(list)
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
                    
                    running_metrics['loss'].append(batch_loss.item() / len(targets))
                    running_metrics['fps'].append(len(targets) / elapsed())
                    output['confidence_scores'].extend(confidence_scores)
                    output['predictions'].extend(predictions)
                    output['targets'].extend(targets)
                    

        print(f'Logging into comet-ml')
        for metric in self.metrics:
            metric_name = metric.__name__
            metric_value = metric(output['targets'], output['predictions'])
            performance_metrics[metric_name] = metric_value
            print(f'{metric_name}: {metric_value}')
            self.logger.log_metric(f'{dataset} {metric_name}', metric_value, step = epoch)

        
        for metric_name, metric_values in running_metrics.items():
            if metric_name == 'fps':
                metric_value = np.mean(metric_values)
                performance_metrics[metric_name] = np.mean(metric_value)
            elif metric_name == 'loss':
                metric_value = np.sum(metric_values)
            performance_metrics[metric_name] = metric_value
            print(f'{metric_name}: {metric_value}')
            self.logger.log_metric(f'{dataset} {metric_name}', metric_value, step = epoch)
        
        return performance_metrics

    def _train_for_all_datasets_in_single_epoch(self, epoch):
        training_performance_metrics = defaultdict(list)
        validation_performance_metrics = defaultdict(list)
        
        print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        for dataset in self.dataloaders.keys():
            single_dataset_performance_metrics = self._train_for_single_dataset(dataset, epoch)
            for metric_name, metric_value in single_dataset_performance_metrics.items():
                if bool(re.search('train', dataset)):
                    training_performance_metrics[metric_name].append(metric_value)
                elif bool(re.search('val', dataset)):
                    validation_performance_metrics[metric_name].append(metric_value)
        self.lr_scheduler.step()
        print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        print(f'#################################################################################')

        for metric_name, metric_value in training_performance_metrics.items():
            average_metric_value = np.mean(metric_value)
            print(f'average training {metric_name}: {average_metric_value}')
            self.logger.log_metric(f'average training {metric_name}', average_metric_value, step = epoch)
            training_performance_metrics[metric_name] = average_metric_value

        for metric_name, metric_value in validation_performance_metrics.items():
            average_metric_value = np.mean(metric_value)
            print(f'average validation {metric_name}: {average_metric_value}')
            self.logger.log_metric(f'average validation {metric_name}', average_metric_value, step = epoch)
            validation_performance_metrics[metric_name] = average_metric_value
        

        return validation_performance_metrics
    
    def _save_model(self, suffix, best_metric : dict, epoch):

        checkpoint_pth = f'{self.checkpoint_path}/{self.model_name}_{suffix}.pth'
        patience_counter = 0

        print(f'Saving training model at epoch {epoch}')
        print(f'Saving training model at {checkpoint_pth}')
        # wandb.save(str(Path.cwd() / checkpoint_pth))

        model_params = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'previous_key': self.logger.get_key()
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
