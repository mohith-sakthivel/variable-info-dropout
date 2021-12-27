import time
import tqdm
import yaml
import pathlib
import argparse
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from vid.models import MLP
from vid.utils.log import Logger
from vid.utils.train import PolyakAveraging, BetaScheduler
from vid.utils.data_loader import CustomSampler, CustomBatchSampler


class ConcatCritic(MLP):

    def forward(self, x, z):
        inp = torch.cat([x, z], dim=1)
        return super().forward(inp)


class NWJ(nn.Module):
    """
    NWJ (Nguyen, Wainwright, and Jordan) estimator
    """

    def __init__(self, label_dim, K):
        super().__init__()
        self._critic = ConcatCritic(label_dim+K, 1, [256,])

    def get_mi_bound(self,  x, z, z_margin=None):
        joint = self._critic(x, z).mean(dim=0)
        if z_margin is not None:
            margin = self._critic(x, z_margin)
        else:
            # change if same label
            margin = self._critic(x, z[torch.randperm(x.shape[0])])
        margin = torch.logsumexp(margin, dim=0) - np.math.log(z.shape[0])
        margin = torch.exp(margin-1)
        return joint - margin


class NWJModel(nn.Module):

    def __init__(self, base_net, K, input_size=28*28, num_classes=10, v='1.0', beta=1e-3, mine_args={}, lr=1e-4,
                 base_net_args={}, use_polyak=True, logdir='.', dropout_p=0.9):
        super().__init__()
        self._K = K
        self._input_size = input_size
        self._num_classes = num_classes
        self._lr = lr
        self._use_polyak = use_polyak
        self.logdir = logdir
        self._dropout_p = dropout_p
        self._v = v

        self._base_net = base_net(input_dim=input_size, output_dim=K, **base_net_args)
        self._model_list = [self._base_net,]
        self._mine_list = []
        main_model = [self._base_net,]

        # setup mine networks
        self._mine = NWJ(10, K)
        self._mine_list.append(self._mine)
        self._mine_args = mine_args
        self._mine_eval_f = NWJ(num_classes, K)
        self._mine_eval_h1 = NWJ(num_classes, int(K/2))
        self._mine_eval_h2 = NWJ(num_classes, int(K/2))
        self._mine_list.extend([self._mine_eval_f, self._mine_eval_h1, self._mine_eval_h2])
        self._beta = BetaScheduler(0, beta, 0) if isinstance(beta, float) else beta
        # setup classifer networks
        self._class = nn.Linear(K, num_classes)
        self._model_list.append(self._class)
        main_model.append(self._class)
        self._class_eval_f = nn.Linear(K, num_classes)
        self._class_eval_h1 = nn.Linear(int(K/2), num_classes)
        self._class_eval_h2 = nn.Linear(int(K/2), num_classes)
        self._model_list.extend([self._class_eval_f, self._class_eval_h1, self._class_eval_h2])

        self.register_buffer('_mask_probs', torch.linspace(1, 1-self._dropout_p, K))
        self._main_model = nn.ModuleList(main_model)
        self._cur_epoch = 0
        self._test_list = ['eval_f', 'eval_h1', 'eval_h2']
        self._test_list.append('classifier')
        self._initialize_weights()
        self._configure_optimizers()
        self._configure_callbacks()
    
    def _initialize_weights(self):
        for (name, param) in itertools.chain(*[module.named_parameters() for module in self._model_list]):
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _configure_optimizers(self):
        model_opt = optim.Adam(self.get_model_parameters(),
                               lr=self._lr, betas=(0.5, 0.999))
        model_sch = {
            'scheduler': optim.lr_scheduler.ExponentialLR(model_opt, gamma=0.97),
            'frequency': 2
        }

        mine_opt = optim.Adam(self.get_mine_parameters(),
                              lr=self._mine_args['est_lr'], betas=(0.5, 0.999))

        mine_sch = {
            'scheduler': optim.lr_scheduler.ExponentialLR(mine_opt, gamma=0.97),
            'frequency': 10
        }

        self.optimizers = [model_opt, mine_opt]
        self.schedulers = [model_sch, mine_sch]

    @property
    def device(self):
        return self._class.weight.device

    def _configure_callbacks(self):
        self._callbacks = []
        if self._use_polyak:
            self._callbacks.append(PolyakAveraging())

    def invoke_callback(self, hook):
        for callback in self._callbacks:
            if hasattr(callback, hook):
                func = getattr(callback, hook)
                func(self)

    def get_model_parameters(self):
        return itertools.chain(*[module.parameters() for module in self._model_list])

    def get_mine_parameters(self):
        return itertools.chain(*[module.parameters() for module in self._mine_list])

    def step_epoch(self):
        self._cur_epoch += 1

    @staticmethod
    def _get_grad_norm(params, device):
        total_grad = torch.zeros([], device=device)
        for param in params:
            total_grad += param.grad.data.norm().square()
        return total_grad.sqrt()

    def _unpack_batch(self, batch):
        batch = [item.to(self.device) for item in batch]
        x, y = batch
        x = x.view(x.shape[0], -1)
        return (x, y)

    def _unpack_margin_batch(self, batch):
        batch = [item.to(self.device) for item in batch]
        x, y = batch
        x = x.view(x.shape[0], -1)
        x, x_margin = torch.chunk(x, chunks=2, dim=0)
        y, y_margin = torch.chunk(y, chunks=2, dim=0)
        return x, y, x_margin, y_margin

    def drop_mask(self, z, normalize=False):
        if z is not None:
            input_probs = torch.ones_like(z) * self._mask_probs
            if normalize:
                return (z * torch.bernoulli(input_probs))/self._mask_probs
            else:
                return z * torch.bernoulli(input_probs)
        else:
            return None

    def _get_embedding(self, x, mc_samples=1):
        z = self._base_net(x)
        if self._base_net.is_stochastic():
            mean, std = z
            z = dist.Independent(dist.Normal(mean, std),
                                 1).rsample([mc_samples])
        else:
            z = z.unsqueeze(dim=0)
        return z

    def _get_train_embedding(self, x):
        z = self._base_net(x)
        z_dist = None
        if self._base_net.is_stochastic():
            mean, std = z
            z_dist = dist.Independent(dist.Normal(mean, std), 1)
            z = z_dist.rsample()
        return z, z_dist
    
    def forward(self, x, mc_samples=1, classifier='main'):
        x = self._get_embedding(x, mc_samples=mc_samples)
        if classifier == 'classifier':
            return self._class(x).mean(dim=0)
        elif classifier == 'eval_f':
            return self._class_eval_f(x).mean(dim=0)
        elif classifier == 'eval_h1':
            return self._class_eval_h1(x[..., :int(self._K/2)]).mean(dim=0)
        elif classifier == 'eval_h2':
            return self._class_eval_h2(x[..., int(self._K/2):]).mean(dim=0)

    def slice(self, tensor, idx, pre=True):
        if tensor is not None:
            if pre:
                return tensor[..., :idx]
            else:
                return tensor[..., idx:]
        return None

    def training_step(self, batch, batch_idx, logger):
        """ Train classifier """
        # unpack data
        if self._mine_args['variant'] == 'marginal':
            x, y, x_margin, _ = self._unpack_margin_batch(batch)
        else:
            x, y, x_margin = *self._unpack_batch(batch), None

        model_opt, mine_opt = self.optimizers
        model_opt.zero_grad()
        mine_opt.zero_grad()
        # get z embedding
        idx = int(self._K/2)
        z, z_dist = self._get_train_embedding(x)
        z_stop = z.detach()
        if x_margin is not None:
            z_margin, _ = self._get_train_embedding(x_margin)
            z_margin_stop = z_margin.detach()
        else:
            z_margin_stop = z_margin = None
        # MI estimators
        one_hot_y = F.one_hot(y, self._num_classes)
        mi_xz = self._mine.get_mi_bound(one_hot_y, self.drop_mask(z), self.drop_mask(z_margin))
        logger.scalar(mi_xz, 'mi_xz', accumulator='train', progbar=True)
        mi_xz_eval_f = self._mine_eval_f.get_mi_bound(one_hot_y, z_stop, z_margin_stop)
        mi_xz_eval_h1 = self._mine_eval_h1.get_mi_bound(one_hot_y, z_stop[..., :idx], self.slice(z_margin_stop, idx, pre=True))
        mi_xz_eval_h2 = self._mine_eval_h2.get_mi_bound(one_hot_y, z_stop[..., idx:], self.slice(z_margin_stop, idx, pre=False))
        # aux_losses
        ce_loss = F.cross_entropy(self._class(z), y)
        logger.scalar(ce_loss, 'ce_loss', accumulator='train', progbar=True)
        ce_loss_eval_f = F.cross_entropy(self._class_eval_f(z_stop), y)
        ce_loss_eval_h1 = F.cross_entropy(self._class_eval_h1(z_stop[..., :idx]), y)
        ce_loss_eval_h2 = F.cross_entropy(self._class_eval_h2(z_stop[..., idx:]), y)
        # Total loss
        beta = self._beta.get(self._cur_epoch)
        total_loss = (-beta * (mi_xz + mi_xz_eval_f + mi_xz_eval_h1 + mi_xz_eval_h2)
                      + ce_loss + ce_loss_eval_f + ce_loss_eval_h1 + ce_loss_eval_h2)
        # log train stats
        if z_dist is not None:
            logger.scalar(z_dist.entropy().mean(), 'z_post_ent', accumulator='train', progbar=True)
            logger.scalar(z_dist.stddev[..., :idx].mean(), 'std_dev_h1', accumulator='train', progbar=False)
            logger.scalar(z_dist.stddev[..., idx:].mean(), 'std_dev_h2', accumulator='train', progbar=False)
        # MI losses
        logger.scalar(mi_xz_eval_f, 'mi_eval_f', accumulator='train', progbar=False)
        logger.scalar(mi_xz_eval_h1, 'mi_eval_h1', accumulator='train', progbar=False)
        logger.scalar(mi_xz_eval_h2, 'mi_eval_h2', accumulator='train', progbar=False)
        logger.scalar(ce_loss_eval_f, 'ce_eval_f', accumulator='train', progbar=False)
        logger.scalar(ce_loss_eval_h1, 'ce_eval_h1', accumulator='train', progbar=False)
        logger.scalar(ce_loss_eval_h2, 'ce_eval_h2', accumulator='train', progbar=False)
        # step optimizer
        total_loss.backward()
        model_opt.step()
        mine_opt.step()
        # Log network gradients
        model_grad_norm = self._get_grad_norm(self._main_model.parameters(), self.device)
        logger.scalar(model_grad_norm, 'model_grad_norm', accumulator='train')

        mine_grad_norm = self._get_grad_norm(self._mine.parameters(), self.device)
        logger.scalar(mine_grad_norm, 'mine_grad_norm', accumulator='train')

    def mine_training_step(self, batch, batch_idx, logger):
        """ Train classifier """
        # unpack data and optimizers
        _, mine_opt = self.optimizers
        if self._mine_args['variant'] == 'marginal':
            x, y, x_margin, _ = self._unpack_margin_batch(batch)
        else:
            x, y, x_margin = *self._unpack_batch(batch), None
        
        mine_opt.zero_grad()
        # get z embedding
        idx = int(self._K/2)
        with torch.no_grad():
            z, _ = self._get_train_embedding(x)
            if x_margin is not None:
                z_margin, _ = self._get_train_embedding(x_margin)
            else:
                z_margin = None
        # MI estimators
        one_hot_y = F.one_hot(y, self._num_classes)
        mi_xz = self._mine.get_mi_bound(one_hot_y, self.drop_mask(z), self.drop_mask(z_margin))
        logger.scalar(mi_xz, 'mi_xz', accumulator='train', progbar=True)
        mi_xz_eval_f = self._mine_eval_f.get_mi_bound(one_hot_y, z, z_margin)
        mi_xz_eval_h1 = self._mine_eval_h1.get_mi_bound(one_hot_y, z[..., :idx], self.slice(z_margin, idx, pre=True))
        mi_xz_eval_h2 = self._mine_eval_h2.get_mi_bound(one_hot_y, z[..., idx:], self.slice(z_margin, idx, pre=False))
        # Total loss
        total_loss = -self._beta.get(self._cur_epoch)*(mi_xz + mi_xz_eval_f + mi_xz_eval_h1 + mi_xz_eval_h2)
        # log train stats
        logger.scalar(mi_xz_eval_f, 'mi_eval_f', accumulator='train', progbar=False)
        logger.scalar(mi_xz_eval_h1, 'mi_eval_h1', accumulator='train', progbar=False)
        logger.scalar(mi_xz_eval_h2, 'mi_eval_h2', accumulator='train', progbar=False)
        # step optimizer
        total_loss.backward()
        mine_opt.step()
        # Log gradient norms
        mine_grad_norm = self._get_grad_norm(self._mine.parameters(), self.device)
        logger.scalar(mine_grad_norm, 'mine_grad_norm', accumulator='train')

    def _get_eval_stats(self, batch, batch_idx, mc_samples=1, classifier='main'):
        stats = {}
        x, y = self._unpack_batch(batch)
        y_pred = self(x, mc_samples, classifier)
        y_pred = torch.argmax(y_pred, dim=1)
        stats['error'] = torch.sum(y != y_pred)/len(y)*100
        return stats

    def evaluation_step(self, batch, batch_idx, logger, mc_samples=1, tag='error', accumulator='test'):
        for classifier in self._test_list:
            stats = self._get_eval_stats(batch, batch_idx, mc_samples, classifier)
            if classifier == 'main':
                logger.scalar(stats['error'], tag, accumulator=accumulator)
            else:
                logger.scalar(stats['error'], classifier +
                            '_'+tag, accumulator=accumulator)


def run(args):

    if args['seed'] is not None:
        torch.manual_seed(args['seed'])

    # setup datasets and dataloaders
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(0.5, 0.5)])

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=data_transforms)
    # use a custom dataloader if z marginals are to be calculated over the true marginal
    if args['model_args']['mine_args']['variant'] == 'marginal':
        # use a custom dataloader if z marginals are to be calculated over the true marginal.
        # each batch contains two 2*batchsize samples [batchsize (for joint) + batchsize (for marginals)]
        sampler = CustomSampler(train_dataset, secondary_replacement=False)
        batch_sampler = CustomBatchSampler(sampler,
                                           batch_size=args['batch_size'],
                                           drop_last=False)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_sampler=batch_sampler,
                                                   num_workers=args['workers'])
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args['batch_size'],
                                                   shuffle=True,
                                                   num_workers=args['workers'])

    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                  transform=data_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args['batch_size'],
                                              shuffle=False,
                                              num_workers=args['workers'])

    # setup logging
    logdir = pathlib.Path(args['logdir'])
    time_stamp = time.strftime("%d-%m-%Y_%H-%M")
    logdir = logdir.joinpath(args['model_id'], '_'.join(
        [args['exp_name'], 's{}'.format(args['seed']), time_stamp]))
    logger = Logger(log_dir=logdir)
    # save experimetn parameters
    with open(logdir.joinpath('hparams.json'), 'w') as out:
        yaml.dump(args, out)
    args['model_args']['logdir'] = logdir

    model = NWJModel(MLP, **args['model_args'])
    print('Using {}...'.format(args['device']))
    model.to(args['device'])

    # Training loop
    model.invoke_callback('on_train_start')
    for epoch in tqdm.trange(1, args['epochs']+1, disable=True):
        model.step_epoch()
        model.train(True)

        for batch_idx, batch in enumerate(tqdm.tqdm(train_loader,
                                                    desc='Model | {}/{} Epochs'.format(
                                                        epoch-1, args['epochs']),
                                                    unit=' batches',
                                                    postfix=logger.get_progbar_desc(),
                                                    leave=False)):
            # Train Model and MINE
            _ = model.training_step(batch, batch_idx, logger)
            model.invoke_callback('on_train_batch_end')
        # Post epoch processing
        _ = logger.scalar_queue_flush('train', epoch)

        for sch in model.schedulers:
            if epoch % sch['frequency'] == 0:
                sch['scheduler'].step()

        # Run validation step
        if (args['validation_freq'] is not None and
            epoch % args['validation_freq'] == 0):
            model.eval()
            # testset used in validation step for observation/study purpose
            for batch_idx, batch in enumerate(test_loader):
                model.evaluation_step(batch, batch_idx, logger, mc_samples=1,
                                      tag='error', accumulator='validation')
                if args['mc_samples'] > 1:
                    model.evaluation_step(batch, batch_idx, logger, mc_samples=args['mc_samples'],
                                          tag='error_mc', accumulator='validation')
       
            _ = logger.scalar_queue_flush('validation', epoch)

    if epoch == args['epochs']:
        model.train(True)
        for sub_epoch in tqdm.trange(1, 11, disable=True):
            for batch_idx, batch in enumerate(tqdm.tqdm(train_loader,
                                                        desc='MINE FineTune | {}/10 Epochs'.format(
                                                            sub_epoch-1),
                                                        unit=' batches',
                                                        leave=False)):
                # Train Model and MINE
                _ = model.mine_training_step(batch, batch_idx, logger)
                model.invoke_callback('on_train_batch_end')
            # Post epoch processing
            _ = logger.scalar_queue_flush('train', epoch+sub_epoch)

    # invoke post training callbacks
    model.invoke_callback('on_train_end')

    # Test model
    model.eval()
    for batch_idx, batch in enumerate(test_loader):
        model.evaluation_step(batch, batch_idx, logger,
                              mc_samples=1, tag='error')
        if args['mc_samples'] > 1:
            model.evaluation_step(batch, batch_idx, logger,
                                  mc_samples=args['mc_samples'], tag='error_mc')
    test_out = logger.scalar_queue_flush('test', epoch)

    print('***************************************************')
    print('Model Test Error: {:.4f}%'.format(test_out['aux_error']))
    if args['mc_samples'] > 1:
        print('Model Test Error ({} sample avg): {:.4f}%'.format(
            args['mc_samples'], test_out['aux_error_mc']))
    print('***************************************************')
    logger.close()


def get_default_args():
    """
    Returns default experiment arguments
    """

    args = {
        'exp_name': 'vid_',
        'seed': 0,
        'model_id': 'default',
        # Trainer args
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 200,
        'logdir': './logs',
        'validation_freq': 1,
        # Dataset args
        'batch_size': 100,
        'workers': 4,
        # Model args
        'model_args': {
            'lr': 1e-4,
            'use_polyak': True,
        }
    }

    args['model_args']['K'] = 8
    args['model_args']['base_net_args'] = {
        'layers': [784, 1024, 1024], 'stochastic': True}
    args['mc_samples'] = 5
    args['model_args']['beta'] = 1e-3
    args['model_args']['mine_args'] = {}
    args['model_args']['mine_args']['estimator'] = 'nwj'
    args['model_args']['mine_args']['est_lr'] = 2e-4
    args['model_args']['mine_args']['variant'] = 'none'
    args['mine_train_steps'] = 1
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Info Flow Experiment')

    parser.add_argument('--v', action='store', default='1.0', type=str)

    parser.add_argument('--exp_name', action='store', type=str,
                        help='Experiment Name')
    parser.add_argument('--seed', action='store', type=int)
    parser.add_argument('--logdir', action='store', type=str,
                        help='Directory to log results')

    parser.add_argument('--K', action='store', type=int)
    parser.add_argument('--epochs', action='store', type=int)
    parser.add_argument('--dropout_p', action='store', type=float)
    parser.add_argument('--beta', action='store', type=float)
    args = parser.parse_args()

    model_args = ['K', 'lr', 'use_polyak', 'beta', 'dropout_p', 'v']
    mine_args = ['estimator', 'est_lr', 'variant']

    exp_args = get_default_args()
    for key, value in args.__dict__.items():
        if value is not None:
            if key in model_args:
                exp_args['model_args'][key] = value
            elif key in mine_args:
                exp_args['model_args']['mine_args'][key] = value
            else:
                exp_args[key] = value

    run(exp_args)
