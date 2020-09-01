#from https://github.com/ajbrock/BigGAN-PyTorch (MIT license) - some modifications
""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).
    Let's go. """
import os
import functools
import math
import numpy as np
use_tqdm=False
if use_tqdm:
    from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
####
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from PyTorchDatasets import CocoAnimals
from PyTorchDatasets import  FFHQ,Celeba
# Import my stuff
import inception_utils
import utils

from PyTorchDatasets import CocoAnimals, FFHQ, Celeba
from fid_score import calculate_fid_given_paths_or_tensor
from torchvision.datasets import ImageFolder
import pickle
from matplotlib import pyplot as plt
from mixup import CutMix
import gc
import sys
from types import ModuleType, FunctionType
from gc import get_referents

####


# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType


def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


# The main training file. Config is a dictionary specifying the configuration of this training run.
#torch.backends.cudnn.benchmark = True

def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def run(config):

    import train_fns

    if config["dataset"]=="coco_animals":
        folders = ['bird','cat','dog','horse','sheep','cow','elephant','monkey','zebra','giraffe']

    # Update the config dict as necessary This is for convenience, to add settings derived from the user-specified configuration into the
    # config-dict (e.g. inferring the number of classes and size of the images from the dataset, passing in a pytorch object for the
    # activation specified as a string)
    config['resolution'] = utils.imsize_dict[config['dataset']]
    print("RESOLUTION: ",config['resolution'])
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    # By default, skip init if resuming training.
    if config['resume']:
        print('Skipping initialization for training resumption...')
        config['skip_init'] = True
    config = utils.update_config_roots(config)
    device = 'cuda'
    # Seed RNG
    utils.seed_rng(config['seed'])
    # Prepare root folders if necessary
    utils.prepare_root(config)
    # Setup cudnn.benchmark for free speed, but only if not more than 4 gpus are used
    if "4" not in config["gpus"]:
        torch.backends.cudnn.benchmark = True
    print(":::::::::::/nCUDNN BENCHMARK", torch.backends.cudnn.benchmark, "::::::::::::::" )
    # Import the model--this line allows us to dynamically select different files.
    model = __import__(config['model'])
    experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
    print('Experiment name is %s' % experiment_name)
    print("::: weights saved at ", '/'.join([config['weights_root'],experiment_name]) )
    # Next, build the model
    keys = sorted(config.keys())
    for k in keys:
        print(k, ": ", config[k])
    G = model.Generator(**config).to(device)

    D = model.Unet_Discriminator(**config).to(device)

    # If using EMA, prepare it
    if config['ema']:
        print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
        G_ema = model.Generator(**{**config, 'skip_init':True,
                                   'no_optim': True}).to(device)
        ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
    else:
        G_ema, ema = None, None
    # FP16?
    if config['G_fp16']:
        print('Casting G to float16...')
        G = G.half()
        if config['ema']:
            G_ema = G_ema.half()
    if config['D_fp16']:
        print('Casting D to fp16...')
        D = D.half()
        # Consider automatically reducing SN_eps?

    GD = model.G_D(G, D, config)
    print(G)
    print(D)
    print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))

    # Prepare noise and randomly sampled label arrays Allow for different batch sizes in G
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    G_batch_size = int(G_batch_size*config["num_G_accumulations"])
    z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])



    # Prepare state dict, which holds things like epoch # and itr #
    state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0,'best_FID': 999999,'config': config}
    # If loading from a pre-trained model, load weights
    if config['resume']:
        print('Loading weights...')
        if config["epoch_id"] !="":
            epoch_id = config["epoch_id"]

        try:
            print("LOADING EMA")
            utils.load_weights(G, D, state_dict,
                            config['weights_root'], experiment_name, config, epoch_id,
                            config['load_weights'] if config['load_weights'] else None,
                            G_ema if config['ema'] else None)
        except:
            print("Ema weight wasn't found, copying G weights to G_ema instead")
            utils.load_weights(G, D, state_dict,
                            config['weights_root'], experiment_name, config, epoch_id,
                            config['load_weights'] if config['load_weights'] else None,
                             None)
            G_ema.load_state_dict(G.state_dict())

        print("loaded weigths")



    # If parallel, parallelize the GD module
    if config['parallel']:
        GD = nn.DataParallel(GD)
    if config['cross_replica']:
        patch_replication_callback(GD)
    # Prepare loggers for stats; metrics holds test metrics, lmetrics holds any desired training metrics.
    test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
    train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
    print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
    test_log = utils.MetricsLogger(test_metrics_fname,
                                 reinitialize=(not config['resume']))
    print('Training Metrics will be saved to {}'.format(train_metrics_fname))
    train_log = utils.MyLogger(train_metrics_fname,
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
    # Write metadata
    utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
    # Prepare data; the Discriminator's batch size is all that needs to be passed to the dataloader, as G doesn't require dataloading. Note
    # that at every loader iteration we pass in enough data to complete a full D iteration (regardless of number of D steps and accumulations)
    D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])



    if config["dataset"]=="FFHQ":

        root = config["data_folder"]
        root_perm =  config["data_folder"]

        transform = transforms.Compose(
            [
                transforms.Scale(config["resolution"]),
                transforms.CenterCrop(config["resolution"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        batch_size = config['batch_size']
        print("rooooot:",root)
        dataset = FFHQ(root = root, transform = transform, batch_size = batch_size*config["num_D_accumulations"], imsize = config["resolution"])
        data_loader = DataLoader(dataset, batch_size, shuffle = True, drop_last = True)
        loaders = [data_loader]

    elif config["dataset"]=="celeba128":

        root =  config["data_folder"] #
        root_perm =  config["data_folder"]
        transform = transforms.Compose(
            [
                transforms.Scale(config["resolution"]),
                transforms.CenterCrop(config["resolution"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        batch_size = config['batch_size']
        dataset = Celeba(root = root, transform = transform, batch_size = batch_size*config["num_D_accumulations"], imsize = config["resolution"])
        data_loader = DataLoader(dataset, batch_size, shuffle = True, drop_last = True)
        loaders = [data_loader]


    elif config["dataset"]=="coco_animals":

        batch_size = config['batch_size']

        transform=transforms.Compose(
                [ transforms.Resize(config["resolution"]),
                    transforms.CenterCrop(config["resolution"]),
                    transforms.RandomHorizontalFlip(),
                    #transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        classes = ['bird','cat','dog','horse','sheep','cow','elephant','monkey','zebra','giraffe']

        root = config["data_folder"]
        root_perm = config["data_folder"]

        dataset = CocoAnimals(root=root, batch_size = batch_size*config["num_D_accumulations"], classes = classes, transform=transform , imsize = config["resolution"])
        data_loader = DataLoader(dataset,batch_size*config["num_D_accumulations"],drop_last=True,num_workers=1)#,shuffle=False)
        loaders = [data_loader]


    print("Loaded ", config["dataset"])
    inception_metrics_dict = {"fid":[],"is_mean": [], "is_std": []}


    # Prepare inception metrics: FID and IS
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'],config['parallel'], config['no_fid'], use_torch=False)

    # Prepare a fixed z & y to see individual sample evolution throghout training
    fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])
    fixed_z.sample_()
    fixed_y.sample_()

    # Loaders are loaded, prepare the training function
    if config['which_train_fn'] == 'GAN':
        train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                                ema, state_dict, config)
    # Else, assume debugging and use the dummy train fn
    else:
        train = train_fns.dummy_training_function()
    # Prepare Sample function for use with inception metrics
    sample = functools.partial(utils.sample,
                          G=(G_ema if config['ema'] and config['use_ema']
                             else G),
                          z_=z_, y_=y_, config=config)



    if config["debug"]:
        loss_steps = 10
    else:
        loss_steps = 100

    print('Beginning training at epoch %d...' % state_dict['epoch'])


    # Train for specified number of epochs, although we mostly track G iterations.
    warmup_epochs = config["warmup_epochs"]


    for epoch in range(state_dict['epoch'], config['num_epochs']):
        if config["progress_bar"]:
            if config['pbar'] == 'mine':
                pbar = utils.progress(loaders[0],displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
            else:
                pbar = tqdm(loaders[0])
        else:
            pbar = loaders[0]

        target_map = None



        for i, batch_data in enumerate(pbar):
            x = batch_data[0]
            y = batch_data[1]
            #H = batch_data[2]


            # Increment the iteration counter
            state_dict['itr'] += 1
            if config["debug"] and state_dict['itr']>config["stop_it"]:
                print("code didn't break :)")
                #exit(0)
                break #better for profiling
            # Make sure G and D are in training mode, just in case they got set to eval For D, which typically doesn't have BN, this shouldn't
            # matter much.
            G.train()
            D.train()
            if config['ema']:
                G_ema.train()
            if config['D_fp16']:
                x, y = x.to(device).half(), y.to(device).view(-1)
            else:
                x, y = x.to(device), y.to(device).view(-1)
            x.requires_grad = False
            y.requires_grad = False



            if config["unet_mixup"]:
                # Here we load cutmix masks for every image in the batch
                n_mixed = int(x.size(0)/config["num_D_accumulations"])
                target_map = torch.cat([CutMix(config["resolution"]).cuda().view(1,1,config["resolution"],config["resolution"]) for _ in range(n_mixed) ],dim=0)


            if config["slow_mixup"] and config["full_batch_mixup"]:
                # r_mixup is the chance that we select a mixed batch instead of
                # a normal batch. This only happens in the setting full_batch_mixup.
                # Otherwise the mixed loss is calculated on top of the normal batch.
                r_mixup = 0.5 * min(1.0, state_dict["epoch"]/warmup_epochs) # r is at most 50%, after reaching warmup_epochs
            elif not config["slow_mixup"] and config["full_batch_mixup"]:
                r_mixup = 0.5
            else:
                r_mixup = 0.0

            metrics = train(x, y, state_dict["epoch"], batch_size , target_map = target_map, r_mixup = r_mixup)


            if (i+1)%200==0:
                # print this just to have some peace of mind that the model is training
                print("alive and well at ", state_dict['itr'])

            if (i+1)%20==0:
                #try:
                train_log.log(itr=int(state_dict['itr']), **metrics)
                #except:
                #    print("ouch")
            # Every sv_log_interval, log singular values
            if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):

                train_log.log(itr=int(state_dict['itr']),
                             **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

          # Save weights and copies as configured at specified interval
            if not (state_dict['itr'] % config['save_every']):

                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()
                    train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                      state_dict, config, experiment_name, sample_only=False)

            go_ahead_and_sample = (not (state_dict['itr'] % config['sample_every']) ) or ( state_dict['itr']<1001 and not (state_dict['itr'] % 100) )

            if go_ahead_and_sample:

                if config['G_eval_mode']:
                    print('Switchin G to eval mode...')
                    G.eval()
                    if config['ema']:
                        G_ema.eval()

                    train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                      state_dict, config, experiment_name, sample_only=True)


                    with torch.no_grad():
                        real_batch = dataset.fixed_batch()
                    train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y,
                                      state_dict, config, experiment_name, sample_only=True, use_real = True, real_batch = real_batch)

                    # also, visualize mixed images and the decoder predicitions
                    if config["unet_mixup"]:
                        with torch.no_grad():

                            n = int(min(target_map.size(0), fixed_z.size(0)/2))
                            which_G = G_ema if config['ema'] and config['use_ema'] else G
                            utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                                                         z_, y_, config['n_classes'],
                                                                         config['num_standing_accumulations'])

                            if config["dataset"]=="coco_animals":
                                real_batch, real_y = dataset.fixed_batch(return_labels = True)

                                fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z[:n], which_G.shared(real_y[:n])))
                                mixed = target_map[:n]*real_batch[:n]+(1-target_map[:n])*fixed_Gz
                                train_fns.save_and_sample(G, D, G_ema, z_[:n], y_[:n], fixed_z[:n], fixed_y[:n],
                                            state_dict, config, experiment_name+"_mix", sample_only=True, use_real = True, real_batch = mixed, mixed=True, target_map = target_map[:n])

                            else:
                                real_batch = dataset.fixed_batch()
                                fixed_Gz = nn.parallel.data_parallel(which_G, (fixed_z[:n], which_G.shared(fixed_z[:n]))) #####shouldnt that be fixed_y?

                                mixed = target_map[:n]*real_batch[:n]+(1-target_map[:n])*fixed_Gz
                                train_fns.save_and_sample(G, D, G_ema, z_[:n], y_[:n], fixed_z[:n], fixed_y[:n],
                                            state_dict, config, experiment_name+"_mix", sample_only=True, use_real = True, real_batch = mixed, mixed=True, target_map = target_map[:n])


          # Test every specified interval
            if not (state_dict['itr'] % config['test_every']):
            #if state_dict['itr'] % 100 == 0:
                if config['G_eval_mode']:
                  print('Switchin G to eval mode...')

                is_mean, is_std , fid = train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics , experiment_name, test_log, moments = "train")
                ###
                #  Here, the bn statistics are updated
                ###
                if  config['accumulate_stats']:
                    print("accumulate stats")
                    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                                                 z_, y_, config['n_classes'], config['num_standing_accumulations'])

                inception_metrics_dict["is_mean"].append((state_dict['itr'] , is_mean ) )
                inception_metrics_dict["is_std"].append((state_dict['itr'] , is_std ) )
                inception_metrics_dict["fid"].append((state_dict['itr'] , fid ) )

            if (i + 1) % loss_steps == 0:
                with open(os.path.join(config["base_root"],"logs/inception_metrics_"+config["random_number_string"]+".p"), "wb") as h:
                    pickle.dump(inception_metrics_dict,h)
                    print("saved FID and IS at", os.path.join(config["base_root"],"logs/inception_metrics_"+config["random_number_string"]+".p") )


        # Increment epoch counter at end of epoch
        state_dict['epoch'] += 1

def main():

    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())

    if config["gpus"] !="":
        os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    random_number_string = str(int(np.random.rand()*1000000)) + "_" + config["id"]
    config["stop_it"] = 99999999999999


    if config["debug"]:
        config["save_every"] = 30
        config["sample_every"] = 20
        config["test_every"] = 20
        config["num_epochs"] = 1
        config["stop_it"] = 35
        config["slow_mixup"] = False

    config["num_gpus"] = len(config["gpus"].replace(",",""))

    config["random_number_string"] = random_number_string
    new_root = os.path.join(config["base_root"],random_number_string)
    if not os.path.isdir(new_root):
        os.makedirs(new_root)
        os.makedirs(os.path.join(new_root, "samples"))
        os.makedirs(os.path.join(new_root, "weights"))
        os.makedirs(os.path.join(new_root, "data"))
        os.makedirs(os.path.join(new_root, "logs"))
        print("created ", new_root)
    config["base_root"] = new_root


    keys = sorted(config.keys())
    print("config")
    for k in keys:
        print(str(k).ljust(30,"."), config[k] )



    run(config)
if __name__ == '__main__':
    main()
