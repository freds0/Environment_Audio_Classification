import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.config = config
        self.data_loader = data_loader
        self.start_epoch = 1

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.checkpoint_dir = config.save_dir
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.metric_ftns = metric_ftns
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.writer = SummaryWriter(config.log_dir)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A average loss.
        """
        self.model.train()
        losses = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data.float())
            loss = self.criterion(output, target)
            losses.append(loss)
            loss.backward()
            self.optimizer.step()

        mean_loss = sum(losses) / len(losses)
        return mean_loss

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A average loss validation
        """
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data.float())
                loss = self.criterion(output, target)
                val_losses.append(loss)
        mean_loss = sum(val_losses) / len(val_losses)
        return mean_loss

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            loss = self._train_epoch(epoch)
            self.writer.add_scalar('Loss/train', loss, epoch)
            if self.do_validation:
                val_loss = self._valid_epoch(epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                #log.update(**{'val_'+k : v for k, v in val_log.items()})

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if epoch % self.save_period == 0:
                best = False
                self._save_checkpoint(epoch, save_best=best)

            print(self._progress(epoch, self.epochs, loss, val_loss))
            self.writer.close()

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint['config']['optimizer']['type'] == self.config['optimizer']['type']:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _progress(self, curr_epoch, total_epoch, loss, val_loss):
        base = '[Epoch {}/{} | Loss: {:.2f} | Val_loss: {:.2f}]'
        return base.format(curr_epoch, total_epoch, loss, val_loss)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids