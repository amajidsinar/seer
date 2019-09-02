from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import numpy as np

__all__ = ['accuracy', 'precision', 'recall', 'f1']

def accuracy(targets, predictions):
    return accuracy_score(targets, predictions)

def precision(targets, predictions):
    return precision_score(targets, predictions)

def recall(targets, predictions):
    return recall_score(targets, predictions)

def f1(targets, predictions):
    return f1_score(targets, predictions)