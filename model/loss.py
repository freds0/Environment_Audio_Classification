import torch.nn.functional as F


def cross_entropy_loss(output, target):
    return F.nn.CrossEntropyLoss(output, target)
