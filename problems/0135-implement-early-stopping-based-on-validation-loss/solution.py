from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    # Your code here
    best_loss=float("inf")
    best_epoch=-1
    wait=0
    for epoch,loss in enumerate(val_losses):
        if loss < ( best_loss- min_delta):
            best_epoch=epoch
            wait=0
            best_loss=loss
        else :
            wait+=1
        if wait>= patience:
            return (epoch,best_epoch)
    return (len(val_losses)-1 , best_epoch)
    pass