"""
Loss functions for the mini neural network framework
"""
from mymicrograd.engine import Value
import math

class Loss:
    """Base loss class"""
    def __call__(self, predictions, targets):
        raise NotImplementedError

class MSELoss(Loss):
    """Mean Squared Error Loss"""
    def __call__(self, predictions, targets):
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]
        
        loss = Value(0.0)
        for pred, target in zip(predictions, targets):
            if not isinstance(target, Value):
                target = Value(target)
            diff = pred - target
            loss = loss + diff * diff
        
        return loss * Value(1.0 / len(predictions))

class CrossEntropyLoss(Loss):
    """Cross Entropy Loss (simplified for binary classification)"""
    def __call__(self, predictions, targets):
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]
        
        loss = Value(0.0)
        for pred, target in zip(predictions, targets):
            if not isinstance(target, Value):
                target = Value(target)
            
            # Apply sigmoid to get probabilities
            sigmoid_pred = Value(1.0) / (Value(1.0) + (pred * Value(-1.0)).exp())
            
            # Cross entropy: -[y*log(p) + (1-y)*log(1-p)]
            # Simplified version to avoid log (which we haven't implemented)
            # Using approximation: -y*p - (1-y)*(1-p) (not exact but for demo)
            loss_term = target * sigmoid_pred * Value(-1.0) - (Value(1.0) - target) * (Value(1.0) - sigmoid_pred)
            loss = loss + loss_term
        
        return loss * Value(1.0 / len(predictions))

class HingeLoss(Loss):
    """Hinge Loss for SVM-style classification"""
    def __call__(self, predictions, targets):
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]
        
        loss = Value(0.0)
        for pred, target in zip(predictions, targets):
            if not isinstance(target, Value):
                target = Value(target)
            
            # Hinge loss: max(0, 1 - y*pred)
            margin = Value(1.0) - target * pred
            hinge = margin.relu()  # max(0, margin)
            loss = loss + hinge
        
        return loss * Value(1.0 / len(predictions))