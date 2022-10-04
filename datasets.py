import os
import sklearn
import scipy
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import scanpy as sc
import anndata as ad
import scsim
import itertools
from pathlib import Path
import joblib
import tensorflow_datasets as tfds

###################################
## Synthetic Tabular Dataset
###################################

def generate_tabular_synth_data(
    n_features,
    spacing=10,
    n_concepts=2,
    dataset_size=10000,
    latent_map=lambda x: x,
    plot=False,
    test_percent=0.2,
    overlap=0,
    seed=0,
):
    np.random.seed(seed)
    latent = np.random.normal(size=(dataset_size, n_features)).astype(np.float32)
    X_train = latent_map(latent)
    c_train = np.zeros((dataset_size, n_concepts), dtype=np.int32)
    ground_truth_concept_masks = np.zeros(shape=(n_concepts, n_features), dtype=np.int32)
    for i in range(n_concepts):
        start = i * spacing
        start = max(start - overlap, 0)
        end = (i + 1) * spacing
        end = min(end + overlap, latent.shape[-1])
        c_train[:, i] = (
            np.sum(latent[:, start:end], axis=-1) > 0
        ).astype(np.int32)
        ground_truth_concept_masks[i, start:end] = 1
    y_train = np.zeros((dataset_size,), dtype=np.int32)
    for i in range(dataset_size):
        bin_str = ''
        for c in c_train[i, :]:
            bin_str += str(c)
        y_train[i] = int(bin_str, 2)
    if plot:
        plt.hist(y_train, bins=len(np.unique(y_train)), weights=np.ones(y_train.shape[0]) / y_train.shape[0])
        plt.show()
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X_train,
        y_train,
        c_train,
        test_size=test_percent,
        random_state=42,
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        c_train,
        c_test,
        ground_truth_concept_masks
    )


###################################
## Synthetic Single-cell dataset
###################################


def generate_synthetic_sc_dataset(
    num_cell_types=10,
    ndoublets=0,
    ngenes=5000,
    ncells=7500,
    act_prog_gene_sizes=[250],
    seed=0,
    n_pcs=16,
    n_neighbors=100,
    test_percent=0.2,
    plot=False,
    min_cells=50,
    min_genes=200,
    min_counts=200,
    dataset_dir=None,
    act_prog_cell_types=None,
):
    if dataset_dir is not None:
        results = []
        broken = False
        for var_name in [
            'X_train',
            'X_test',
            'y_train',
            'y_test',
            'c_train',
            'c_test',
            'ground_truth_concept_masks',
        ]:
            path = os.path.join(dataset_dir, var_name + f"_seed_{seed}.npy")
            if os.path.exists(path):
                results.append(np.load(path))
            else:
                broken = True
                break
        adata_path = os.path.join(dataset_dir, f"adata_seed_{seed}.joblib")
        if (not broken) and os.path.exists(adata_path):
            adata = joblib.load(adata_path)
            results.append(adata)
            if plot:
                sc.pl.pca_variance_ratio(adata, log=True)
                sc.pl.umap(
                    adata,
                    color='cell_type_str_viz',
                    use_raw=True,
                    ncols=3,
                    title="Identity GEPs"
                )
                sc.pl.umap(
                    adata,
                    color='viz_label_str',
                    use_raw=True,
                    ncols=3,
                    title='Task Label Annotations',
                )
                sc.pl.umap(
                    adata,
                    color='activity_program_str',
                    use_raw=True,
                    ncols=3,
                    title="Activity Programs",
                )
                sc.pl.umap(
                    adata,
                    color='activity_program_str_viz',
                    use_raw=True,
                    ncols=3,
                    title="Activity GEP Status"
                )
            return tuple(results)
        
    simulator = scsim.scsim(
        ngenes=ngenes,
        ncells=ncells,
        n_cell_types=num_cell_types,
        ndoublets=ndoublets,

        libloc=7.64,
        libscale=0.78,
        mean_rate=7.68,
        mean_shape=0.34,
        expoutprob=0.00286,
        expoutloc=6.15,
        expoutscale=0.49,
        diffexpprob=0.025,
        diffexpdownprob=0.,
        diffexploc=1.0,
        diffexpscale=1.0,
        bcv_dispersion=0.448,
        bcv_dof=22.087,

        act_prog_gene_sizes=act_prog_gene_sizes,
        act_prog_down_prob=0.,
        act_prog_de_loc=1.0,
        act_prog_de_scale=1,
        act_prog_cell_frac=0.3,
        act_prog_cell_types=act_prog_cell_types or [
            list(range(1, num_cell_types + 1, 1))
            for _ in act_prog_gene_sizes
        ],
        min_act_prog_usage=0.1,
        max_act_prog_usage=0.7,
        seed=seed,
    )
    simulator.simulate()
    
    simulator.cellparams['n_counts'] = np.sum(simulator.counts.to_numpy(), axis=-1)
    simulator.cellparams['cell_type_str'] = [
        f'cell_type_{i}' for i in simulator.cellparams['cell_type']
    ]
    
    adata = ad.AnnData(simulator.counts, obs=simulator.cellparams)
    # And add a label based on the activity and identity programs generates
    adata.obs['label'] = [0 if np.sum(x) == 0 else 1 for x in adata.obs['has_act_program']]
    adata.obs['label_str'] = [f'class_{0 if np.sum(x) == 0 else 1}' for x in adata.obs['has_act_program']]
    adata.obs['activity_program_str'] = [str(x) for x in adata.obs['has_act_program']]
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=200)
    kept_gene_mask, num_per_gene = sc.pp.filter_genes(adata, min_cells=min_cells, inplace=False)
    sc.pp.filter_genes(adata, min_cells=min_cells, inplace=True)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    
    # Make combination labels
    used_combos = set()
    for i in range(adata.shape[0]):
        used_combos.add((adata.obs.cell_type[i], tuple(adata.obs.has_act_program[i])))
    combination_label_map = {}
    inv_combination_label_map = {}
    for cell_type in range(1, simulator.n_cell_types + 1):
        for activity_program_combo in itertools.product(*[[0, 1] for _ in simulator.act_prog_gene_sizes]):
            key = (cell_type, tuple(activity_program_combo))
            if key in used_combos:
                combination_label_map[key] = len(combination_label_map)
                inv_combination_label_map[combination_label_map[key]] = key
    adata.obs['combination_label'] = [
        combination_label_map[(adata.obs.cell_type[i], tuple(adata.obs.has_act_program[i]))]
        for i in range(adata.shape[0])
    ]
    adata.obs['combination_label_str'] = [f'comb_label_{x}' for x in adata.obs['combination_label']]
    adata.obs['viz_label_str'] = [f'{x+1}' for x in adata.obs['combination_label']]

    # Mean and variance normalize the genes
    sc.pp.scale(adata, zero_center=False)
    adata.raw = adata.copy()
    
    # Run PCA
    sc.pp.pca(adata)
    
    if plot:
        # Make a scree plot to determine number of PCs to use for UMAP
        sc.pl.pca_variance_ratio(adata, log=True)
    
    # Construct the nearest neighbor graph for UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    
    # Run UMAP
    sc.tl.umap(adata)
    
    # Plot the UMAP with some cannonical marker genes to see that the apparent clustering makes sense
    adata.obs['cell_type_str_viz'] = [f'Identity GEP {i}' for i in adata.obs['cell_type']]
    if plot:
        sc.pl.umap(
            adata,
            color='cell_type_str_viz',
            use_raw=True,
            ncols=3,
            title="Identity GEPs"
        )
    
    # with rc_context({'figure.figsize': (8, 6)}):
    # Plot the UMAP with some cannonical marker genes to see that the apparent clustering makes sense
    if plot:
        sc.pl.umap(
            adata,
            color='viz_label_str',
            use_raw=True,
            ncols=3,
            title='Task Label Annotations',
            gene_symbols=[str(i + 1) for i in range(len(combination_label_map))],
        )
    
    # Plot the UMAP with some cannonical marker genes to see that the apparent clustering makes sense
    if plot:
        sc.pl.umap(
            adata,
            color='activity_program_str',
            use_raw=True,
            ncols=3,
            title="Activity Programs",
        )
    
    # Plot the UMAP with some cannonical marker genes to see that the apparent clustering makes sense
    adata.obs['activity_program_str_viz'] = ['True'if x[0] else 'False' for x in adata.obs['has_act_program']]
    if plot:
        sc.pl.umap(
            adata,
            color='activity_program_str_viz',
            use_raw=True,
            ncols=3,
            title="Activity GEP Status"
        )
    
    # And produce the training data we will all love and use
    X_train = adata.to_df().to_numpy()
    y_train = adata.obs["combination_label"].to_numpy()
    # The concepts will be a concatenation of the cell type and
    # the activation of each activity program
    c_train = np.concatenate(
        [
            tf.one_hot(adata.obs["cell_type"].to_numpy() - 1, num_cell_types, axis=-1),
            np.array([
                x for x in adata.obs['has_act_program'].to_numpy()
            ]),
        ],
        axis=-1,
    )
    
    # To Construct ground truth masks first look over all identity GEPs
    ground_truth_concept_masks = []
    for prog_idx in range(num_cell_types):
        ground_truth_concept_masks.append(np.array([
            1 if simulator.geneparams[f'cell_type_{prog_idx + 1}_gene_selection'][j]
            else 0 for j in range(simulator.ngenes)
            if kept_gene_mask[j]
        ]))
    # Then over all activity GEPs
    for prog_idx in range(len(simulator.activity_program_genes)):
        ground_truth_concept_masks.append(np.array([
            1 if j in simulator.activity_program_genes[prog_idx] else 0 for j in range(simulator.ngenes)
            if kept_gene_mask[j]
        ]))
    
    # Split it up into test/train splits
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X_train,
        y_train,
        c_train,
        test_size=test_percent,
        random_state=42,
    )
    
    if dataset_dir is not None:
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        for var_name, var in [
            ('X_train', X_train),
            ('X_test', X_test),
            ('y_train', y_train),
            ('y_test', y_test),
            ('c_train', c_train),
            ('c_test', c_test),
            ('ground_truth_concept_masks', ground_truth_concept_masks),
        ]:
            np.save(
                os.path.join(dataset_dir, var_name + f"_seed_{seed}.npy"),
                var,
            )
        joblib.dump(
            adata,
            os.path.join(dataset_dir, f"adata_seed_{seed}.joblib")
        )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        c_train,
        c_test,
        ground_truth_concept_masks,
        adata
    )


###################################
## PBMC
###################################


def generate_pbmc_data(
    seed=0,
    test_percent=0.2,
    plot=False,
    min_cells=200, #50,
    min_genes=200,
    min_counts=500, #200,
    n_pcs=32,
    n_neighbors=100,
    dataset_dir="data/pbmc/",
):
    counts = scipy.sparse.load_npz(os.path.join(dataset_dir, "X.npz")).toarray()
    labels = scipy.sparse.load_npz(os.path.join(dataset_dir, "y.npz")).toarray()
    labels = np.reshape(labels, -1)
    obs = pd.DataFrame(data=labels,columns=["label"])
    print(obs)
    adata = ad.AnnData(counts, obs=obs)
    # And add a label based on the activity and identity programs generates
    adata.obs['label_str'] = [f'class_{x}' for x in adata.obs['label']]
    
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=200)
    kept_gene_mask, num_per_gene = sc.pp.filter_genes(adata, min_cells=min_cells, inplace=False)
    sc.pp.filter_genes(adata, min_cells=min_cells, inplace=True)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    
    # Mean and variance normalize the genes
    sc.pp.scale(adata, zero_center=False)
    adata.raw = adata.copy()
    
    # Run PCA
    sc.pp.pca(adata)
    
    if plot:
        # Make a scree plot to determine number of PCs to use for UMAP
        sc.pl.pca_variance_ratio(adata, log=True)
    
    # Construct the nearest neighbor graph for UMAP
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    
    # Run UMAP
    sc.tl.umap(adata)
    
    # Plot the UMAP with some cannonical marker genes to see that the apparent clustering makes sense
    if plot:
        sc.pl.umap(
            adata,
            color='label_str',
            use_raw=True,
            ncols=3,
            title='Task Label Annotations',
        )
    
    # And produce the training data we will all love and use
    X_train = adata.to_df().to_numpy()
    y_train = adata.obs["label"].to_numpy().astype(np.int32)

    # Split it up into test/train splits
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=test_percent,
        random_state=seed,
    )
    
    return (X_train, X_test, y_train, y_test)

###################################
## Forest Cover Dataset
###################################


def generate_forest_cover_data(
    dataset_dir="data/covtype.csv",
    test_percent=0.2,
    seed=0,
):
    data = pd.read_csv("data/covtype.csv") 
    # Preview the first 5 lines of the loaded data 
    X = data.to_numpy()
    X_train, y_train = X[:, :-1], X[:, -1]
    y_train = (y_train - 1).astype(np.int32)
    X_train = X_train.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train,
        y_train,
        test_size=test_percent,
        random_state=seed,
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


###################################
## Higgs dataset
###################################

def generate_higgs_data(
    include_high_level=True,
    test_percent=0.2,
    seed=42,
    load_batch_size=8096,
    dataset_dir="data/higgs_numpy"
):
    if os.path.exists(os.path.join(dataset_dir, "X_train.npy")):
        X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
        X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
        y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
        y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))
        c_train = np.load(os.path.join(dataset_dir, "c_train.npy"))
        c_test = np.load(os.path.join(dataset_dir, "c_test.npy"))
        # Redo the sampling to respect the seed selection and also
        # any possible changes in test percents!
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
            np.concatenate([X_train, X_test], axis=0),
            np.concatenate([y_train, y_test], axis=0),
            np.concatenate([c_train, c_test], axis=0),
            test_size=test_percent,
            random_state=seed,
        )
    else:
        # Else let's generate the data from scratch
        high_level_feats = sorted([
            'm_bb',
            'm_jj',
            'm_jjj',
            'm_jlv',
            'm_lv',
            'm_wbb',
            'm_wwbb',
        ])
        prev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        # Ignote GPU to avoid flooding it with data
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        print("Loading Higgs dataset...")
        ds = tfds.load('higgs', split='train', shuffle_files=True)
        print("\tDone")
        ds = ds.batch(load_batch_size)
        Xs = [None for _ in range(len(ds))]
        ys = [None for _ in range(len(ds))]
        Cs = [None for _ in range(len(ds))]

        print("Generating Higgs matrices...")
        ds_numpy = tfds.as_numpy(ds)
        print("\tTurned to numpy")
        for i, x in enumerate(ds_numpy):
            print(f'{(i + 1)/len(ds) * 100:.2f}% (size {i + 1})', end="\r")
            feats = sorted(x.keys())
            tensor = []
            concepts = []
            for feat_name in feats:
                if feat_name == "class_label":
                    y = x[feat_name]
                elif feat_name in high_level_feats:
                    concepts.append(x[feat_name])
                else:
                    tensor.append(x[feat_name])
            ys[i] = np.array(y)
            Xs[i] = np.array(tensor).T
            Cs[i] = np.array(concepts).T
        print("\tDone!")
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        C = np.concatenate(Cs, axis=0)
        if prev is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
            X,
            y,
            C,
            test_size=test_percent,
            random_state=seed,
        )
        if dataset_dir:
            Path(dataset_dir).mkdir(parents=True, exist_ok=True)
            for var_name, var in [
                ('X_train', X_train),
                ('X_test', X_test),
                ('y_train', y_train),
                ('y_test', y_test),
                ('c_train', c_train),
                ('c_test', c_test),
            ]:
                np.save(
                    os.path.join(dataset_dir, var_name + f".npy"),
                    var,
                )
    if not include_high_level:
        return X_train, X_test, y_train.astype(np.int32), y_test.astype(np.int32), c_train, c_test
    
    # Else let's put everything back into the same array
    X_train = np.concatenate([X_train, c_train], axis=-1)
    X_test = np.concatenate([X_test, c_test], axis=-1)
    return X_train, X_test, y_train, y_test



###################################
## Standard Datasets
###################################

def generate_tabular_synth_linear_data(seed):
    n_ground_truth_concepts = 2
    extra_hyperparameters = {
        'n_ground_truth_concepts': n_ground_truth_concepts,
    }
    data = generate_tabular_synth_data(
        dataset_size=15000,
        n_features=100,
        spacing=5,
        n_concepts=2,
        latent_map=lambda x: x,
        plot=False,
    )
    return data, extra_hyperparameters

def generate_tabular_synth_nonlinear_data(seed):
    n_ground_truth_concepts = 2
    extra_hyperparameters = {
        'n_ground_truth_concepts': n_ground_truth_concepts,
    }
    data = generate_tabular_synth_data(
        dataset_size=15000,
        n_features=100,
        spacing=5,
        n_concepts=2,
        latent_map=lambda x: np.sin(x) + x,
        plot=False,
    )
    return data, extra_hyperparameters

def generate_tabular_synth_nonlinear_large_data(seed):
    n_ground_truth_concepts = 5
    extra_hyperparameters = {
        'n_ground_truth_concepts': n_ground_truth_concepts,
    }
    data = generate_tabular_synth_data(
        dataset_size=15000,
        n_features=500,
        spacing=20,
        overlap=5,
        n_concepts=n_ground_truth_concepts,
        latent_map=lambda x: np.sin(x) + x,
        plot=False,
        seed=seed,
    )
    return data, extra_hyperparameters


def generate_synth_sc_data(
    seed,
    num_cell_types=10,
    n_act_progs=1,
    include_adata=False,
    ngenes=5000,
    ncells=7500,
    test_percent=0.2,
    ndoublets=0,
    dataset_dir="data/sc_synth",
    n_pcs=16,
    n_neighbors=100,
    min_genes=200,
    min_counts=200,
    min_cells=50,
    act_prog_size=250,
    act_prog_cell_types=None,
):
    prev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    # Ignote GPU to avoid flooding it with data
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    result = generate_synthetic_sc_dataset(
        num_cell_types=num_cell_types,
        ndoublets=ndoublets,
        ngenes=ngenes,
        ncells=ncells,
        act_prog_gene_sizes=[act_prog_size for _ in range(n_act_progs)],
        seed=(42 + seed),
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        test_percent=test_percent,
        plot=False,
        min_cells=min_cells,
        min_genes=min_genes,
        min_counts=min_counts,
        dataset_dir=dataset_dir,
        act_prog_cell_types=act_prog_cell_types,
    )
    if prev is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev
    else:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    adata = result[-1]
    extra_hyperparameters = {
        'avg_group_size': int(np.mean(np.unique(adata.obs['cell_type'], return_counts=True)[1])),
        'num_cell_types': num_cell_types,
        'n_act_progs': n_act_progs,
        'n_ground_truth_concepts': n_act_progs + num_cell_types,
        'start_ncells': ncells,
        'start_ngenes': ngenes,
    }
        
    if include_adata:
        return result, extra_hyperparameters
    return result[:-1], extra_hyperparameters
