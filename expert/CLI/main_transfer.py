from expert.src.model import Model
import torch
from expert.src.onn_dataset import ONNDataSet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
from expert.src.utils import read_genus_abu, read_labels, load_otlg, zero_weight_unk, parse_otlg, get_dmax, transfer_weights
import pandas as pd, numpy as np
import os
from expert.CLI.CLI_utils import find_pkg_resource

def transfer(cfg, args):
    # Read data
    X, idx = read_genus_abu(args.input)
    Y = read_labels(args.labels, shuffle_idx=idx, dmax=get_dmax(args.labels))
    print('Reordering labels and samples...')
    IDs = sorted(list(set(X.index.to_list()).intersection(Y[0].index.to_list())))
    X = X.loc[IDs, :]
    Y = [y.loc[IDs, :] for y in Y]
    print('Total matched samples:', sum(X.index == Y[0].index))

    # Basic configurations
    phylogeny = pd.read_csv(find_pkg_resource('resources/phylogeny.csv'), index_col=0)
    new_mapper = cfg.getboolean('transfer', 'new_mapper')
    reuse_levels = cfg.get('transfer', 'reuse_levels')
    finetune_eps = cfg.getint('transfer', 'finetune_epochs')
    finetune_lr = cfg.getfloat('transfer', 'finetune_lr')
    epochs = cfg.getint('transfer', 'epochs')
    lr = cfg.getfloat('transfer', 'lr')
    min_lr = cfg.getfloat('transfer', 'min_lr')
    reduce_patience = cfg.getint('transfer', 'reduce_patience')
    stop_patience = cfg.getint('transfer', 'stop_patience')
    label_smoothing = cfg.getfloat('transfer', 'label_smoothing')
    batch_size = cfg.getint('transfer', 'batch_size')
    stopper = EarlyStopping(monitor='val_loss', patience=stop_patience)
 
    # if args.log:
    #     logger = CSVLogger(filename=args.log)
    #     ft_logger = CSVLogger(filename=args.log, append=True)
    #     callbacks.append(logger)
    #     ft_callbacks.append(ft_logger)
    dropout_rate = args.dropout_rate

    # Build EXPERT model
    ontology = load_otlg(args.otlg)
    _, layer_units = parse_otlg(ontology)
    base_model = Model(phylogeny=phylogeny, restore_from=args.model)
    init_model = Model(phylogeny=phylogeny, ontology=ontology, dropout_rate=dropout_rate)

    # All transferred blocks and layers will be set to be non-trainable automatically.
    model = transfer_weights(base_model, init_model, reuse_levels)
    print('Total correct samples: {}?{}'.format(sum(X.index == Y[0].index), Y[0].shape[0]))

    # Feature encoding and standardization
    X = torch.from_numpy(X.to_numpy())
    X = model.encoder(X)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    if args.update_statistics:
        model.update_statistics(mean=X.mean(axis=0), std=X.std(axis=0))
    X = model.standardize(X)

    # Sample weight "zero" to mask unknown samples' contribution to loss
    sample_weight = [zero_weight_unk(y=y, sample_weight=np.ones(y.shape[0])) for i, y in enumerate(Y)]
    Y = [y.drop(columns=['Unknown']) for y in Y]
    Y = [torch.from_numpy(y.to_numpy()).float() for y in Y]
    
    # Create dataset
    dataset = ONNDataSet(X, Y)
    val_size = int(len(X)*args.val_split)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Train EXPERT model
    print('Training using optimizer with lr={}...'.format(lr))
    trainer = Trainer(accelerator='auto', max_epochs=epochs, callbacks=[stopper])
    model.configure_lr(pretrain_lr=lr, reduce_patience=reduce_patience)
    trainer.fit(model, train_loader, val_loader)

    if args.finetune:
        finetune_eps += stopper.stopped_epoch
        print('Fine-tuning using optimizer with lr={}...'.format(finetune_lr))
        model.base.requires_grad_(True)
        for layer in range(model.n_layers):
            model.spec_inters[layer].requires_grad_(True)
            model.spec_integs[layer].requires_grad_(True)
            model.spec_outputs[layer].requires_grad_(True)
        trainer = Trainer(accelerator='auto', max_epochs=finetune_eps, callbacks=[stopper])
        model.lr = finetune_lr
        trainer.fit(model, train_loader, val_loader)

    # Save EXPERT model
    model.save_blocks(args.output)
