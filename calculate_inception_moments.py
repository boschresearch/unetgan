#from https://github.com/ajbrock/BigGAN-PyTorch (MIT license) - some modifications
''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.

 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser
from torchvision import datasets, transforms, utils
from PyTorchDatasets import CocoAnimals3  as CocoAnimals
from PyTorchDatasets import FFHQ, LSUN, CelebaHQ, Celeba
from torch.utils.data import DataLoader

def prepare_parser():
  usage = 'Calculate and store inception metrics.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)')
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  return parser


def run(config):
  # Get loader
  config['drop_last'] = False

  if config["dataset"]=="FFHQ":
      imsize = 256

      root =  os.path.join(os.environ["SSD"],"images256x256")
      root_perm = os.path.join(os.environ["SSD"],"images256x256")

      transform = transforms.Compose(
          [
              transforms.Scale(imsize),
              transforms.CenterCrop(imsize),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
          ]
      )

      batch_size = 100#config['batch_size']
      dataset = FFHQ(root = root, transform = transform, test_mode = False)
      data_loader = DataLoader(dataset, batch_size, shuffle = True, drop_last = True)
      loaders = [data_loader]

  elif config["dataset"]=="coco":

      imsize = 128
      batch_size = config['batch_size']

      transform=transforms.Compose(
              [ transforms.Resize(imsize),
                  transforms.CenterCrop(imsize),
                  #transforms.RandomHorizontalFlip(),
                  #transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
              ])

      classes = ['bird','cat','dog','horse','sheep','cow','elephant','monkey','zebra','giraffe']


      root = None
      root_perm = None
      dataset = CocoAnimals(root=root, batch_size = batch_size, classes = classes, transform=transform, masks=False , return_all = True, test_mode = False, imsize=imsize)
      data_loader = DataLoader(dataset,batch_size ,drop_last=True,num_workers=1, shuffle = True)#,shuffle=False)
      loaders = [data_loader]

  else:
      loaders = utils.get_data_loaders(**config)

  # Load inception net
  net = inception_utils.load_inception_net(parallel=config['parallel'])
  pool, logits, labels = [], [], []
  device = 'cuda'
  used_samples = 0
  for e in range(2):
    for i, batch_data  in enumerate(tqdm(loaders[0])):

      x = batch_data[0]
      y = batch_data[1]
      x = x.to(device)
      with torch.no_grad():
        pool_val, logits_val = net(x)
        pool += [np.asarray(pool_val.cpu())]
        logits += [np.asarray(F.softmax(logits_val, 1).cpu())]
        labels += [np.asarray(y.cpu())]

      used_samples += x.size(0)
      if used_samples >=50000:
        break

  pool, logits, labels = [np.concatenate(item, 0) for item in [pool, logits, labels]]
  # uncomment to save pool, logits, and labels to disk
  # print('Saving pool, logits, and labels to disk...')
  # np.savez(config['dataset']+'_inception_activations.npz',
  #           {'pool': pool, 'logits': logits, 'labels': labels})
  # Calculate inception metrics and report them
  print('Calculating inception metrics...')
  IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
  print('Training data from dataset %s has IS of %5.5f +/- %5.5f' % (config['dataset'], IS_mean, IS_std))
  # Prepare mu and sigma, save to disk. Remove "hdf5" by default
  # (the FID code also knows to strip "hdf5")
  print('Calculating means and covariances...')
  print(pool.shape)
  mu, sigma = np.mean(pool, axis=0), np.cov(pool, rowvar=False)
  print('Saving calculated means and covariances to disk...')
  np.savez(config['dataset'].strip('_hdf5')+'_inception_moments.npz', **{'mu' : mu, 'sigma' : sigma})

def main():
  # parse command line
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':
    main()
