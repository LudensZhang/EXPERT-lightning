from expert.src.model import Model
import torch
from expert.src.onn_dataset import ONNDataSet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split, TensorDataset
from expert.src.utils import read_genus_abu, read_labels, load_otlg, zero_weight_unk, parse_otlg, get_dmax
import pandas as pd, numpy as np
from expert.CLI.CLI_utils import find_pkg_resource

def train(cfg, args):     
    # Reading data
    X, idx = read_genus_abu(args.input)
    if args.rg:
        Y = [read_labels(args.labels, shuffle_idx=idx, regression=True)]
    else:
        Y = read_labels(args.labels, shuffle_idx=idx, dmax=get_dmax(args.labels))
    print('Reordering labels and samples...')
    IDs = sorted(list(set(X.index.to_list()).intersection(Y[0].index.to_list())))
    X = X.loc[IDs, :]
    Y = [y.loc[IDs, :] for y in Y]
    print('Total matched samples:', sum(X.index == Y[0].index))

    # Reading basic configurations from config file
    pretrain_ep = cfg.getint('train', 'pretrain_ep')
    pretrain_lr = cfg.getfloat('train', 'pretrain_lr')

    lr = cfg.getfloat('train', 'lr')
    epochs = cfg.getint('train', 'epochs')
    reduce_patience = cfg.getint('train', 'reduce_patience')
    stop_patience = cfg.getint('train', 'stop_patience')
    batch_size = cfg.getint('train', 'batch_size')
    
    stopper = EarlyStopping(monitor='val_loss', patience=stop_patience)
    
    # if args.log:
    #     pretrain_logger = CSVLogger(filename=args.log)
    #     logger = CSVLogger(filename=args.log)
    #     pretrain_callbacks.append(pretrain_logger)
    #     callbacks.append(logger)
    phylogeny = pd.read_csv(find_pkg_resource('resources/phylogeny.csv'), index_col=0)
    dropout_rate = args.dropout_rate

    # Calculate sample weight for each layer, assign 0 weight for sample with 0 labels
    #sample_weight = [zero_weight_unk(y=y, sample_weight=compute_sample_weight(class_weight='balanced',
    #                                                                          y=y.to_numpy().argmax(axis=1)))
    #                 for i, y in enumerate(Y_train)]

    # Build the model
    if args.rg:
        model = Model(phylogeny=phylogeny, dropout_rate=dropout_rate, regression=True)
    else:
        ontology = load_otlg(args.otlg)
        _, layer_units = parse_otlg(ontology)
        model = Model(phylogeny=phylogeny,ontology=ontology, dropout_rate=dropout_rate)

    # Feature encoding and standardization
    X = torch.from_numpy(X.to_numpy())
    X = model.encoder(X)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    print('N. NaN in input features:', np.isnan(X).sum())
    model.update_statistics(mean=X.mean(axis=0), std=X.std(axis=0))
    X = model.standardize(X)

    #------------------------------- SELECTIVE LEARNING-----------------------------------------------
    # Sample weight "zero" to mask unknown samples' contribution to loss
    Y = Y[0] if args.rg else [y.drop(columns=['Unknown']) for y in Y]
    Y = torch.from_numpy(Y.to_numpy()).float() if args.rg else [torch.from_numpy(y.to_numpy()).float() for y in Y]
    
    # Create dataset
    dataset = TensorDataset(X, Y) if args.rg else ONNDataSet(X, Y)
    val_size = int(len(X)*args.val_split)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Train EXPERT model
    print('Pre-training using Adam with lr={}...'.format(pretrain_lr))
    trainer = Trainer(accelerator='auto', max_epochs=pretrain_ep, callbacks=[stopper])
    model.configure_lr(pretrain_lr=pretrain_lr, reduce_patience=reduce_patience)
    trainer.fit(model, train_loader, val_loader)
    
    print('Training using Adam with lr={}...'.format(lr))
    trainer = Trainer(accelerator='auto', max_epochs=epochs, callbacks=[stopper])
    model.lr = lr
    trainer.fit(model, train_loader, val_loader)

    # Save the EXPERT model
    model.save_blocks(args.output)
