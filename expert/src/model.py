import os
import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import OrderedDict
import numpy as np
import pandas as pd
from livingTree import SuperTree
from tqdm import tqdm

def relu_init(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def sig_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def load_otlg(path):
    otlg = SuperTree().from_pickle(path)
    return otlg

def parse_otlg(ontology):
    labels = OrderedDict([(layer, label) for layer, label in ontology.get_ids_by_level().items()
                          if layer > 0])
    layer_units = [len(label) for layer, label in labels.items()]
    return labels, layer_units


class Model(LightningModule):

    def __init__(self, 
                 phylogeny, 
                 ontology=None, 
                 restore_from=None, 
                 regression=False,
                 dropout_rate=0, 
                 open_set=False):
        super(Model, self).__init__()
        self.expand_dims = torch.unsqueeze
        self.concat = torch.cat
        self.dropout = nn.Dropout(dropout_rate)
        self.rg = regression
        self.open_set = open_set
        self.num_features = phylogeny.shape[0]*phylogeny.shape[1]
        if ontology:
            self.ontology = ontology
            self.labels, self.layer_units = parse_otlg(self.ontology)
            self.n_layers = len(self.layer_units)
            self.statistics = pd.DataFrame(index=range(self.num_features), columns=['mean', 'std'], dtype=np.float32)
            self.base = self.init_base_block(num_features=self.num_features) 
            self.spec_inters = nn.ModuleList([self.init_inter_block(index=layer, name='l{}_inter'.format(layer+2), n_units=n_units)
                                for layer, n_units in enumerate(self.layer_units)])
            self.spec_integs = nn.ModuleList([self.init_integ_module(index=layer, name='l{}_integration'.format(layer + 2), n_units=n_units)
                                for layer, n_units in enumerate(self.layer_units)])
            self.spec_outputs = nn.ModuleList([self.init_output_module(index=layer, name='l{}o'.format(layer + 2), n_units=n_units)
                                 for layer, n_units in enumerate(self.layer_units)])
        elif restore_from:
            self.__restore_from(restore_from)
            self.n_layers = len(self.spec_outputs)
        else:
            raise ValueError('Please given correct model path to restore, '
                             'or specify layer_units to build model from scratch.')
        self.encoder = self.init_encoder_block(phylogeny)
        self.spec_postprocs = nn.ModuleList([self.init_post_proc_module(name='l{}'.format(layer + 2)) for layer in range(self.n_layers)])

    def forward(self, x):
        x = x.view(-1, self.num_features)
        x = self.dropout(x)
        base = self.base(x)
        inter_logits = [self.spec_inters[i](base) for i in range(self.n_layers)]
        integ_logits = []
        for layer in range(self.n_layers):
            if layer == 0:
                integ_logits.append(self.spec_integs[layer](inter_logits[layer]))
            else:
                logits = torch.cat([integ_logits[layer - 1], inter_logits[layer]], dim=1)
                integ_logits.append(self.spec_integs[layer](logits))
        out_probas = [self.spec_outputs[i](integ_logits[i]) for i in range(self.n_layers)]
        return out_probas
    
    def configure_lr(self, pretrain_lr, reduce_patience):
        '''
        Configure learning rate and reduce on plateau scheduler.
        '''
        self.lr = pretrain_lr
        self.reduce_patience = reduce_patience  
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        print(self(batch))
        return self(batch)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.MSELoss() if self.rg else nn.BCEWithLogitsLoss()
        y_hat = self(x)
        
        if not self.rg:
            y = torch.split(y, self.layer_units, dim=1)
            loss = [loss(y_hat[i], y[i]) for i in range(self.n_layers)]
            loss = torch.stack(loss)
            weight = torch.tensor(self.layer_units)/torch.tensor(self.layer_units).sum()
            weight = weight.to(self.device)
            loss = (weight*loss).sum()
        else:
            loss = loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.MSELoss() if self.rg else nn.BCEWithLogitsLoss()
        y_hat = self(x)
        
        if not self.rg:
            y = torch.split(y, self.layer_units, dim=1)
            loss = [loss(y_hat[i], y[i]) for i in range(self.n_layers)]
            loss = torch.stack(loss)
            weight = torch.tensor(self.layer_units)/torch.tensor(self.layer_units).sum()
            weight = weight.to(self.device)
            loss = (weight*loss).sum()
        else:
            loss = loss(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min', patience=self.reduce_patience, factor=0.5),
                     'monitor': 'val_loss'}
        return [optimizer], [scheduler]
    
    def cal_proba(self, logits):
        if self.n_layers == 1:
            logits = [logits]
        
        contrib = [self.spec_postprocs[i](logits[i]) for i in tqdm(range(self.n_layers))]
        
        return contrib
    
    def save_blocks(self, path):
        inters_dir = self.__pthjoin(path, 'inters')
        integs_dir = self.__pthjoin(path, 'integs')
        outputs_dir = self.__pthjoin(path, 'outputs')
        for dir in [path, inters_dir, integs_dir, outputs_dir]:
            if not os.path.isdir(dir):
                os.mkdir(dir)
        torch.save(self.base.state_dict(), self.__pthjoin(path, 'base'))
        self.ontology.to_pickle(self.__pthjoin(path, 'ontology.pkl'))
        self.statistics.to_csv(self.__pthjoin(path, 'statistics.csv'))
        for layer in range(self.n_layers):
            torch.save(self.spec_inters[layer].state_dict(), self.__pthjoin(inters_dir, str(layer)))
            torch.save(self.spec_integs[layer].state_dict(), self.__pthjoin(integs_dir, str(layer)))
            torch.save(self.spec_outputs[layer].state_dict(), self.__pthjoin(outputs_dir, str(layer)))

    def __restore_from(self, path):
        otlg_dir = self.__pthjoin(path, 'ontology.pkl')
        base_dir = self.__pthjoin(path, 'base')
        inters_dir = self.__pthjoin(path, 'inters')
        integs_dir = self.__pthjoin(path, 'integs')
        outputs_dir = self.__pthjoin(path, 'outputs')
        inter_dirs = [self.__pthjoin(inters_dir, i) for i in sorted(os.listdir(inters_dir), key=lambda x: int(x))]
        integ_dirs = [self.__pthjoin(integs_dir, i) for i in sorted(os.listdir(integs_dir), key=lambda x: int(x))]
        output_dirs = [self.__pthjoin(outputs_dir, i) for i in sorted(os.listdir(outputs_dir), key=lambda x: int(x))]
        self.ontology = load_otlg(otlg_dir)
        self.statistics = pd.read_csv(self.__pthjoin(path, 'statistics.csv'), index_col=0, dtype=np.float32)
        self.labels, self.layer_units = parse_otlg(self.ontology)
        self.base = self.init_base_block(num_features=self.num_features)
        self.base.load_state_dict(torch.load(base_dir))
        self.spec_inters = nn.ModuleList([self.init_inter_block(index=layer, name='l{}_inter'.format(layer+2), n_units=n_units)
                                for layer, n_units in enumerate(self.layer_units)])
        self.spec_integs = nn.ModuleList([self.init_integ_module(index=layer, name='l{}_integration'.format(layer + 2), n_units=n_units)
                                for layer, n_units in enumerate(self.layer_units)])
        self.spec_outputs = nn.ModuleList([self.init_output_module(index=layer, name='l{}o'.format(layer + 2), n_units=n_units)
                                 for layer, n_units in enumerate(self.layer_units)])
        for layer in range(len(self.layer_units)):
            self.spec_inters[layer].load_state_dict(torch.load(inter_dirs[layer]))
            self.spec_integs[layer].load_state_dict(torch.load(integ_dirs[layer]))
            self.spec_outputs[layer].load_state_dict(torch.load(output_dirs[layer]))

    def init_encoder_block(self, phylogeny):
        block = Encoder(phylogeny)
        return block

    def init_base_block(self, num_features):
        block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 2**10),
            nn.ReLU(),
            nn.Linear(2**10, 2**9),
            nn.ReLU()
        )
        block.apply(relu_init)
        return block

    def init_inter_block(self, index, name, n_units):
        k = index
        block = nn.Sequential(
            nn.Linear(2**9, 8*n_units),
            nn.ReLU(),
            nn.Linear(8*n_units, 4*n_units),
            nn.ReLU(),
            nn.Linear(4*n_units, 2*n_units),
            nn.ReLU()
        )
        block.apply(relu_init)
        return block

    def init_integ_module(self, index, name, n_units):
        if index == 0:
            input_size = 2*n_units
        else:
            input_size = 2*n_units + 3*self.layer_units[index-1]
            
        block = nn.Sequential(
            nn.Linear(input_size, 3*n_units),
            nn.Tanh()
        )
        block.apply(sig_init)
        return block

    def init_output_module(self, index, name, n_units):
        input_size = 3*n_units           
        block = nn.Sequential(
            nn.Linear(input_size, n_units)
        )
        block.apply(sig_init)
        return block

    def init_post_proc_module(self, name):
        def cal_unknown(x):
            unkn =  1 - x.sum(dim=1, keepdim=True)
            cat_unkn = torch.cat([x, unknown], dim=1)
            return cat_unkn.div(cat_unkn.sum(dim=1, keepdim=True), dim=0)
        
        if self.open_set:
            block = nn.Sequential(
                nn.sigmoid(),
                cal_unknown
            )
        else:
            block = nn.Sequential(
                nn.Softmax(dim=1)
            )
        return block

    def update_statistics(self, mean, std):
        stats = self.statistics.copy()
        stats.loc[:, 'mean'] = mean.tolist()
        stats.loc[:, 'std'] = std.tolist()
        self.statistics = stats
        print(self.statistics)

    def standardize(self, X):
        mean = torch.from_numpy(self.statistics['mean'].to_numpy())
        std = torch.from_numpy(self.statistics['std'].to_numpy())
        return X.sub(mean).div(std + 1e-8)

    def __pthjoin(self, pth1, pth2):
        return os.path.join(pth1, pth2)


class Encoder(nn.Module):

    def __init__(self, phylogeny, name=None):
        super(Encoder, self).__init__()
        self.ranks = phylogeny.columns.to_list()[:-1]
        self.W = {rank: self.get_W(phylogeny[rank]) for rank in self.ranks}
        self.expand_dims = torch.unsqueeze

    def get_W(self, taxons):
        cols = taxons.to_numpy().reshape(taxons.shape[0], 1)
        rows = taxons.to_numpy().reshape(1, taxons.shape[0])
        return torch.tensor((rows == cols).astype(np.float32))

    def forward(self, inputs):
        F_genus = inputs
        F_ranks = [self.expand_dims(torch.matmul(F_genus, self.W[rank]), dim=2) for rank in self.ranks] + \
                  [self.expand_dims(F_genus, dim=2)]
        outputs = torch.cat(F_ranks, dim=2)
        return outputs