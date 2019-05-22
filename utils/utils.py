# utils.py
import torch
from torch.autograd import Variable
import time


def check_accuracy(model, loader, dtype):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x_var = Variable(x.type(dtype))
            scores = model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    return 100 * acc


class Trainier:
    def __init__(self, model, optimizer, dtype):
        self.model = model
        self.optimizer = optimizer
        self.dtype = dtype
        self.val_accs = []
        self.train_accs = []

    def train(self, num_epochs, loss_fn, train_loader, valid_loader, print_every):
        tic = time.time()
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for t, (x, y) in enumerate(train_loader):
                x_var = Variable(x.type(self.dtype))
                y_var = Variable(y.type(self.dtype).long())

                scores = self.model(x_var)
                
                loss = loss_fn(scores, y_var)
                total_loss += loss.data

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % print_every == 0:        
                val_acc = check_accuracy(self.model, valid_loader, self.dtype)
                train_acc = check_accuracy(self.model, train_loader, self.dtype)
                toc = time.time()
                print('Epoch %d/%d => Time: %.2fsec, Train avg loss: %.4f, train acc: %.2f%%, val acc: %.2f%%' % 
                      (epoch + 1, num_epochs, toc-tic, total_loss/(t+1), train_acc, val_acc))
                tic = toc
            self.train_accs += [train_acc]
            self.val_accs += [val_acc]
