"""
Early stopping mechanism for training.
"""


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, min_delta=0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            verbose (bool): Print messages when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        """
        Check if early stopping criteria is met.

        Args:
            val_loss: Current validation loss
            epoch: Current epoch number

        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  ⚠️  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'\n  🛑 Early stopping triggered! Best epoch was {self.best_epoch + 1}')
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
