import torch

class EarlyStopping:
    """
    当验证集损失在一段时间内不再改善时，提前停止训练。
    """
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存当前最佳模型权重
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"Validation loss decreased. Saving model...")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        """ 训练结束后加载最佳权重 """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model