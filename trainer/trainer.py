import numpy as np
import torch
import os
#from torchvision.utils import make_grid
#from base import BaseTrainer
#from utils import inf_loop, MetricTracker
#from logger import TensorboardWriter
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        #super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        #self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
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
        #self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
        self.writer = SummaryWriter(config.log_dir)
        #self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        #self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        #self.train_metrics.reset()
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
            #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            #self.train_metrics.update('loss', loss.item())
            #for met in self.metric_ftns:
            #    self.train_metrics.update(met.__name__, met(output, target))

            #if batch_idx % self.log_step == 0:
            #    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            #        epoch,
            #        self._progress(batch_idx),
            #        loss.item()))
            #    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            #if batch_idx == self.len_epoch:
            #    break
        #log = self.train_metrics.result()

        #if self.do_validation:
        #    val_log = self._valid_epoch(epoch)
        #    log.update(**{'val_'+k : v for k, v in val_log.items()})

        #if self.lr_scheduler is not None:
        #    self.lr_scheduler.step()
        #return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        #self.valid_metrics.reset()
        val_losses = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data.float())
                loss = self.criterion(output, target)
                val_losses.append(loss)
                #self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                #self.valid_metrics.update('loss', loss.item())
                #for met in self.metric_ftns:
                #    self.valid_metrics.update(met.__name__, met(output, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        mean_loss = sum(val_losses) / len(val_losses)
        return mean_loss
        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        #return self.valid_metrics.result()

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

            '''
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break
            '''
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
            #'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        #self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            #self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        #self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        #self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        '''
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        '''
        self.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint['config']['optimizer']['type'] == self.config['optimizer']['type']:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
    """
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    """
    def _progress(self, curr_epoch, total_epoch, loss, val_loss):
        base = '[Epoch {}/{} | Loss: {:.2f} | Val_loss: {:.2f}]'
        return base.format(curr_epoch, total_epoch, loss, val_loss)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            #self.logger.warning("Warning: There\'s no GPU available on this machine,"
            #                    "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            #self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
            #                    "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids