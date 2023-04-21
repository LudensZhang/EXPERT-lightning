from expert.src.model import Model
import os
import  pandas as pd
from expert.CLI.CLI_utils import find_pkg_resource
import torch
        
def search(cfg, args):

    # Read data
    X = pd.read_hdf(args.input, key='genus').T
    sampleIDs = X.index
    X = torch.from_numpy(X.to_numpy())
    phylogeny = pd.read_csv(find_pkg_resource('resources/phylogeny.csv'), index_col=0)

    # Build EXPERT model
    model = Model(phylogeny=phylogeny, restore_from=args.model,
                  open_set=args.measure_unknown, regression=args.rg)
    X = model.encoder(X).reshape(X.shape[0], X.shape[1] * phylogeny.shape[1])
    X = model.standardize(X)
    

    # Calculate source contribution
    contrib_arrs = model(X)
    if args.rg:
        result = pd.DataFrame(contrib_arrs.detach().numpy(), index=sampleIDs, columns=['y_predicted'])
        os.makedirs(args.output, exist_ok=True)
        result.to_csv(os.path.join(args.output, 'predicted.csv'))
        return
    
    contrib_arrs = model.cal_proba(contrib_arrs)
    
    if model.n_layers == 1:
        contrib_arrs = [contrib_arrs]
    labels = model.labels
    if model.open_set:
        contrib_layers = {
            'layer-' + str(i + 2): pd.DataFrame(contrib_arrs[i].detach().numpy(), index=sampleIDs, columns=labels[i + 1] + ['Unknown'])
            for i, key in enumerate(labels.keys())}
    else:
        contrib_layers = {
            'layer-' + str(i + 2): pd.DataFrame(contrib_arrs[i].detach().numpy(), index=sampleIDs, columns=labels[i + 1])
            for i, key in enumerate(labels.keys())}

    for layer, contrib in contrib_layers.items():
        if not os.path.isdir(args.output):
            os.mkdir(args.output)
        contrib.to_csv(os.path.join(args.output, layer+'.csv'))
