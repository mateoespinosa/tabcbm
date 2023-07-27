'''
Adapted from https://github.com/dylkot/scsim

All credit given to Kotliar et al.
'''


import pandas as pd
import numpy as np

class scsim:
    def __init__(
        self,
        ngenes=10000,
        ncells=100,
        seed=42,
        mean_rate=.3,
        mean_shape=.6,
        libloc=11,
        libscale=0.2,
        expoutprob=.05,
        expoutloc=4,
        expoutscale=0.5,
        n_cell_types=1,
        diffexpprob=None,
        diff_exp_prob_genes=None,
        diffexpdownprob=.5,
        diffexploc=.1,
        diffexpscale=.4,
        bcv_dispersion=.1,
        bcv_dof=60,
        ndoublets=0,
        cell_type_probs=None,


        act_prog_gene_sizes=None,
        act_prog_genes=None,
        act_prog_down_prob=None,
        act_prog_de_loc=1.0,
        act_prog_de_scale=1.0,
        act_prog_cell_types=None,
        act_prog_cell_frac=None,
        min_act_prog_usage=.2,
        max_act_prog_usage=.8,
    ):

        if (
            ((diffexpprob is None) and (diff_exp_prob_genes is None)) or
            ((diff_exp_prob_genes is not None) and (diffexpprob is not None))
        ):
            raise ValueError('Either diff_exp_prob_genes or diffexpprob need to be provided')

        if diff_exp_prob_genes is not None:
            if len(diff_exp_prob_genes) != n_cell_types:
                raise ValueError(
                    f'Expected as many lists in diff_exp_prob_genes (found {len(diff_exp_prob_genes)}) '
                    f'as the number of cell types ({n_cell_types})'
                )
            diffexpprob = [
                [1 if i in cell_type_genes else 0 for i in range(ngenes)]
                for cell_type_genes in diff_exp_prob_genes
            ]
        else:
            diffexpprob = [
                [diffexpprob for i in range(ngenes)]
                for _ in range(n_cell_types)
            ]

        if (
            (act_prog_gene_sizes is not None) and (act_prog_genes is not None)
        ):
            raise ValueError(
                'Expected at most one of act_prog_gene_sizes or act_prog_genes to be provided.'
            )
        if (act_prog_genes is not None):
            act_prog_gene_sizes = [len(x) for x in act_prog_genes]


        self.ngenes = ngenes
        self.ncells = ncells
        self.seed = seed
        self.mean_rate = mean_rate
        self.mean_shape = mean_shape
        self.libloc = libloc
        self.libscale = libscale
        self.expoutprob = expoutprob
        self.expoutloc = expoutloc
        self.expoutscale = expoutscale
        self.n_cell_types = n_cell_types
        self.diffexpprob = diffexpprob
        self.diffexpdownprob = diffexpdownprob
        self.diffexploc = diffexploc
        self.diffexpscale = diffexpscale
        self.bcv_dispersion = bcv_dispersion
        self.bcv_dof = bcv_dof
        self.ndoublets = ndoublets
        self.init_ncells = ncells + ndoublets


        # How many randomly selected genes will be involved in each activity program
        self.act_prog_gene_sizes = act_prog_gene_sizes
        if (self.act_prog_gene_sizes is not None) and (not isinstance(self.act_prog_gene_sizes, (list, np.ndarray))):
            self.act_prog_gene_sizes = [self.act_prog_gene_sizes]

        # Already select which genes will be involved in each activity program
        if self.act_prog_gene_sizes is not None:
            if act_prog_genes is not None:
                self.activity_program_genes = act_prog_genes
            else:
                self.activity_program_genes = [None for _ in self.act_prog_gene_sizes]
                for i, n_genes_in_act_prog in enumerate(self.act_prog_gene_sizes):
                    self.activity_program_genes[i] = np.sort(
                        np.random.choice(self.ngenes, n_genes_in_act_prog, replace=False)
                    )
        # The probability that a given activity program is downregulated
        self.act_prog_down_prob = act_prog_down_prob
        if self.act_prog_gene_sizes is not None:
            if not isinstance(self.act_prog_down_prob, (list, np.ndarray)):
                self.act_prog_down_prob = [self.act_prog_down_prob for _ in self.act_prog_gene_sizes]
            elif len(self.act_prog_down_prob) != len(self.act_prog_gene_sizes):
                raise ValueError(
                    'Expected the probability of downregulating an activity program to be a scalar or an iterable with '
                    f'as many elements as provided activity programs. Instead we obtained {self.act_prog_down_prob} '
                    f'while we were also given {len(self.act_prog_gene_sizes)} activity programs to simulate.'
                )

        self.act_prog_de_loc = act_prog_de_loc
        if self.act_prog_gene_sizes is not None:
            if not isinstance(self.act_prog_de_loc, (list, np.ndarray)):
                self.act_prog_de_loc = [self.act_prog_de_loc for _ in self.act_prog_gene_sizes]
            elif len(self.act_prog_de_loc) != len(self.act_prog_gene_sizes):
                raise ValueError(
                    'Expected the mean DE parameter an activity program to be a scalar or an iterable with '
                    f'as many elements as provided activity programs. Instead we obtained {self.act_prog_de_loc} '
                    f'while we were also given {len(self.act_prog_gene_sizes)} activity programs to simulate.'
                )

        self.act_prog_de_scale = act_prog_de_scale
        if self.act_prog_gene_sizes is not None:
            if not isinstance(self.act_prog_de_scale, (list, np.ndarray)):
                self.act_prog_de_scale = [self.act_prog_de_scale for _ in self.act_prog_gene_sizes]
            elif len(self.act_prog_de_scale) != len(self.act_prog_gene_sizes):
                raise ValueError(
                    'Expected the scale DE parameter an activity program to be a scalar or an iterable with '
                    f'as many elements as provided activity programs. Instead we obtained {self.act_prog_de_scale} '
                    f'while we were also given {len(self.act_prog_gene_sizes)} activity programs to simulate.'
                )

        # Decide which cell types can be involved in each activity program. If not given then we assume that
        # activity programs can be distributed across all cell types
        self.act_prog_cell_types = act_prog_cell_types
        if (self.act_prog_gene_sizes is not None) and (self.act_prog_cell_types is None):
            ## The program is active in all cell types
            self.act_prog_cell_types = [
                np.arange(1, self.n_cell_types + 1) for _ in self.act_prog_gene_sizes
            ]
        if self.act_prog_gene_sizes is not None:
            if not isinstance(self.act_prog_cell_types[0], (list, np.ndarray)):
                self.act_prog_cell_types = [np.array(self.act_prog_cell_types) for _ in self.act_prog_gene_sizes]
            elif len(self.act_prog_cell_types) != len(self.act_prog_gene_sizes):
                raise ValueError(
                    'Expected the set of cell types involved with each activity prgram to be a single list of integers or a '
                    f'list with as many lists as provided activity programs. Instead we obtained {self.act_prog_cell_types} '
                    f'while we were also given {len(self.act_prog_gene_sizes)} activity programs to simulate.'
                )

        # What percentage of cells will contain each activation program
        self.act_prog_cell_frac = act_prog_cell_frac
        if self.act_prog_gene_sizes is not None:
            if not isinstance(self.act_prog_cell_frac, (list, np.ndarray)):
                self.act_prog_cell_frac = [self.act_prog_cell_frac for _ in self.act_prog_gene_sizes]
            elif len(self.act_prog_cell_frac) != len(self.act_prog_gene_sizes):
                raise ValueError(
                    'Expected the number of activity program cell frequencies to be a scalar or an iterable with '
                    f'as many elements as provided activity programs. Instead we obtained {self.act_prog_cell_frac} '
                    f'while we were also given {len(self.act_prog_gene_sizes)} activity programs to simulate.'
                )

        self.min_act_prog_usage = min_act_prog_usage
        if self.act_prog_gene_sizes is not None:
            if not isinstance(self.min_act_prog_usage, (list, np.ndarray)):
                self.min_act_prog_usage = [self.min_act_prog_usage for _ in self.act_prog_gene_sizes]
            elif len(self.min_act_prog_usage) != len(self.act_prog_gene_sizes):
                raise ValueError(
                    'Expected the min activity program usuage parameter to be a scalar or an iterable with '
                    f'as many elements as provided activity programs. Instead we obtained {self.min_act_prog_usage} '
                    f'while we were also given {len(self.act_prog_gene_sizes)} activity programs to simulate.'
                )
        self.max_act_prog_usage = max_act_prog_usage
        if self.act_prog_gene_sizes is not None:
            if not isinstance(self.max_act_prog_usage, (list, np.ndarray)):
                self.max_act_prog_usage = [self.max_act_prog_usage for _ in self.act_prog_gene_sizes]
            elif len(self.max_act_prog_usage) != len(self.act_prog_gene_sizes):
                raise ValueError(
                    'Expected the max activity program usuage parameter to be a scalar or an iterable with '
                    f'as many elements as provided activity programs. Instead we obtained {self.max_act_prog_usage} '
                    f'while we were also given {len(self.act_prog_gene_sizes)} activity programs to simulate.'
                )

        if cell_type_probs is None:
            # Then we will assume a uniform distribution over all possible cell types
            self.cell_type_probs = [1/self.n_cell_types] * self.n_cell_types
        elif (len(cell_type_probs) == self.n_cell_types) and (np.abs(np.sum(cell_type_probs) - 1) < (1e-6)):
            # Else, it is a a valid probability distribution
            self.cell_type_probs = cell_type_probs
        else:
            raise ValueError(f'Invalid cell type probabilities {cell_type_probs}')


    def simulate(self):
        np.random.seed(self.seed)
        print('Simulating cells')
        self.cellparams = self.get_cell_params()
        print('Simulating gene params')
        self.geneparams = self.get_gene_params()

        if (self.act_prog_gene_sizes is not None) and (len(self.act_prog_gene_sizes) > 0):
            print('Simulating activity programs')
            self.simulate_activity_programs()

        print('Simulating DE')
        self.sim_cell_type_DE()

        print('Simulating cell-gene means')
        self.cell_gene_mean = self.get_cell_gene_means()
        if self.ndoublets > 0:
            print('Simulating doublets')
            self.simulate_doublets()

        print('Adjusting means')
        self.adjust_means_bcv()
        print('Simulating counts')
        self.simulate_counts()

    def simulate_counts(self):
        '''Sample read counts for each gene x cell from Poisson distribution
        using the variance-trend adjusted updated_mean value'''
        self.counts = pd.DataFrame(
            np.random.poisson(lam=self.updated_mean),
            index=self.cellnames,
            columns=self.genenames,
        )

    def adjust_means_bcv(self):
        '''Adjust cell_gene_mean to follow a mean-variance trend relationship'''
        self.bcv = self.bcv_dispersion + (1 / np.sqrt(self.cell_gene_mean))
        chisamp = np.random.chisquare(self.bcv_dof, size=self.ngenes)
        self.bcv = self.bcv*np.sqrt(self.bcv_dof / chisamp)
        self.updated_mean = np.random.gamma(
            shape=1/(self.bcv**2),
            scale=self.cell_gene_mean*(self.bcv**2),
        )
        self.bcv = pd.DataFrame(self.bcv, index=self.cellnames, columns=self.genenames)
        self.updated_mean = pd.DataFrame(
            self.updated_mean,
            index=self.cellnames,
            columns=self.genenames,
        )


    def simulate_doublets(self):
        ## Select doublet cells and determine the second cell to merge with
        d_ind = sorted(
            np.random.choice(self.ncells, self.ndoublets, replace=False)
        )
        d_ind = [f'Cell{x + 1}' for x in d_ind]
        self.cellparams['is_doublet'] = False
        self.cellparams.loc[d_ind, 'is_doublet'] = True
        extra_ind = self.cellparams.index[-self.ndoublets:]
        group2 = self.cellparams.ix[extra_ind, 'group'].values
        self.cellparams['group2'] = -1
        self.cellparams.loc[d_ind, 'group2'] = group2

        ## update the cell-gene means for the doublets while preserving the
        ## same library size
        dmean = self.cell_gene_mean.loc[d_ind,:].values
        dmultiplier = 0.5 / dmean.sum(axis=1)
        dmean = np.multiply(dmean, dmultiplier[:, np.newaxis])
        omean = self.cell_gene_mean.loc[extra_ind,:].values
        omultiplier = 0.5 / omean.sum(axis=1)
        omean = np.multiply(omean, omultiplier[:,np.newaxis])
        newmean = dmean + omean
        libsize = self.cellparams.loc[d_ind, 'libsize'].values
        newmean = np.multiply(newmean, libsize[:,np.newaxis])
        self.cell_gene_mean.loc[d_ind,:] = newmean
        ## remove extra doublet cells from the data structures
        self.cell_gene_mean.drop(extra_ind, axis=0, inplace=True)
        self.cellparams.drop(extra_ind, axis=0, inplace=True)
        self.cellnames = self.cellnames[0:self.ncells]


    def get_cell_gene_means(self):
        '''Calculate each gene's mean expression for each cell while adjusting
        for the library size'''


        cell_type_gene_mean = self.geneparams.loc[
            :,
            [x for x in self.geneparams.columns if ('_gene_mean' in x) and ('cell_type' in x)]
        ].T.astype(float)
        cell_type_gene_mean = cell_type_gene_mean.div(cell_type_gene_mean.sum(axis=1), axis=0)
        ind = self.cellparams['cell_type'].apply(lambda x: f'cell_type_{x}_gene_mean')

        if (self.act_prog_gene_sizes is None) or (len(self.act_prog_gene_sizes) == 0):
            cell_gene_mean = cell_type_gene_mean.loc[ind, :].astype(float)
            cell_gene_mean.index = self.cellparams.index
        else:
            no_act_prog_cells = np.sum(
                np.array(list(map(list, self.cellparams['has_act_program'].values))),
                axis=-1,
            ) == 0
            has_act_prog_cells = np.sum(
                np.array(list(map(list, self.cellparams['has_act_program'].values))),
                axis=-1,
            ) != 0

            print('   - Getting mean for activity program carrying cells')
            act_prog_cell_mean = cell_type_gene_mean.loc[ind[has_act_prog_cells], :]
            act_prog_cell_mean.index = ind.index[has_act_prog_cells]
            act_prog_cell_mean = act_prog_cell_mean.multiply(
                1 - self.cellparams.loc[has_act_prog_cells, 'act_program_usage'],
                axis=0,
            )

            act_prog_mean = self.geneparams.loc[:,['act_prog_gene_mean']]
            act_prog_mean = act_prog_mean.div(act_prog_mean.sum(axis=0), axis=1)
            act_prog_usage = self.cellparams.loc[act_prog_cell_mean.index, ['act_program_usage']]
            act_prog_usage.columns = ['act_prog_gene_mean']
            act_prog_cell_mean += act_prog_usage.dot(act_prog_mean.T)
            # TODO (verify this): if multiple identity programs use the same gene, then its resulting mean will be the
            #                     mean of the expected means of all identity programs
            act_prog_cell_mean = act_prog_cell_mean.applymap(np.mean)
            act_prog_cell_mean = act_prog_cell_mean.astype(float)

            print('   - Getting mean for non activity program carrying cells')
            no_act_prog_cell_mean = cell_type_gene_mean.loc[ind[no_act_prog_cells], :]
            no_act_prog_cell_mean.index = ind.index[no_act_prog_cells]

            cell_gene_mean = pd.concat([no_act_prog_cell_mean, act_prog_cell_mean], axis=0, verify_integrity=True)
            # Sort them so that they are in the same order as before
            cell_gene_mean.sort_index(inplace=True, key=lambda x: [int(y[len('Cell'):]) for y in x])

        print('   - Normalizing by cell libsize')
        normfac = (self.cellparams['libsize'] / cell_gene_mean.sum(axis=1)).values
        for col in cell_gene_mean.columns:
            cell_gene_mean[col] = cell_gene_mean[col].values * normfac
        return cell_gene_mean


    def get_gene_params(self):
        '''Sample each genes mean expression from a gamma distribution as
        well as identifying outlier genes with expression drawn from a
        log-normal distribution'''
        base_gene_mean = np.random.gamma(
            shape=self.mean_shape,
            scale=1/self.mean_rate,
            size=self.ngenes,
        )

        # Figure out which cells will be considered to be outliers
        is_outlier = np.random.choice(
            [True, False],
            size=self.ngenes,
            p=[self.expoutprob, 1 - self.expoutprob],
        )
        outlier_ratio = np.ones(shape=self.ngenes)
        outliers = np.random.lognormal(
            mean=self.expoutloc,
            sigma=self.expoutscale,
            size=is_outlier.sum(),
        )
        outlier_ratio[is_outlier] = outliers
        gene_mean = base_gene_mean.copy()
        median = np.median(base_gene_mean)
        gene_mean[is_outlier] = outliers * median
        self.genenames = [f'Gene{i}' for i in range(1, self.ngenes + 1)]
        geneparams = pd.DataFrame(
            [base_gene_mean, is_outlier, outlier_ratio, gene_mean],
            index=['BaseGeneMean', 'is_outlier', 'outlier_ratio', 'gene_mean'],
            columns=self.genenames,
        ).T
        return geneparams


    def get_cell_params(self):
        '''Sample cell type identities and library sizes'''
        cell_type_ids = self.simulate_cell_types()
        libsize = np.random.lognormal(
            mean=self.libloc,
            sigma=self.libscale,
            size=self.init_ncells,
        )
        self.cellnames = [f'Cell{i}' for i in range(1, self.init_ncells + 1)]
        cellparams = pd.DataFrame(
            [cell_type_ids, libsize],
            index=['cell_type', 'libsize'],
            columns=self.cellnames,
        ).T
        cellparams['cell_type'] = cellparams['cell_type'].astype(int)
        return cellparams


    def simulate_activity_programs(self):
        ## Simulate expression of all requested activity programs
        self.geneparams = self.geneparams.join(
            pd.DataFrame(
                {
                    'act_prog_gene': [
                        np.zeros((len(self.act_prog_gene_sizes),), dtype=np.int32)
                        for _ in range(self.ngenes)
                    ]
                },
                index=[f'Gene{i}' for i in range(1, self.geneparams.shape[0] + 1)]
            ),
        )
        self.geneparams = self.geneparams.join(
            pd.DataFrame(
                {
                    'act_prog_gene_mean': [
                        np.zeros((len(self.act_prog_gene_sizes),), dtype=np.float32)
                        for _ in range(self.ngenes)
                    ]
                },
                index=[f'Gene{i}' for i in range(1, self.geneparams.shape[0] + 1)]
            ),
        )


        self.cellparams = self.cellparams.join(
            pd.DataFrame(
                {
                    'has_act_program': [
                        np.zeros((len(self.act_prog_gene_sizes),), dtype=np.int32)
                        for _ in range(self.ncells)
                    ]
                },
                index=[f'Cell{i}' for i in range(1, self.cellparams.shape[0] + 1)]
            ),
        )
        self.cellparams = self.cellparams.join(
            pd.DataFrame(
                {
                    'act_program_usage': [
                        np.zeros((len(self.act_prog_gene_sizes),), dtype=np.float32)
                        for _ in range(self.ncells)
                    ]
                },
                index=[f'Cell{i}' for i in range(1, self.cellparams.shape[0] + 1)]
            ),
        )

        for act_prog_idx, n_genes_in_act_prog in enumerate(self.act_prog_gene_sizes):
            act_prog_genes = self.activity_program_genes[act_prog_idx]
            for vec in self.geneparams.loc[
                [f'Gene{i + 1}' for i in act_prog_genes],
                'act_prog_gene',
            ]:
                vec[act_prog_idx] = 1
            DEratio = np.random.lognormal(
                mean=self.act_prog_de_loc[act_prog_idx],
                sigma=self.act_prog_de_scale[act_prog_idx],
                size=n_genes_in_act_prog,
            )
            DEratio[DEratio < 1] = 1 / DEratio[DEratio < 1]
            is_downregulated = np.random.choice(
                [True, False],
                size=len(DEratio),
                p=[
                    self.act_prog_down_prob[act_prog_idx],
                    1 - self.act_prog_down_prob[act_prog_idx],
                ],
            )
            DEratio[is_downregulated] = 1 / DEratio[is_downregulated]
            all_DE_ratio = np.ones(self.ngenes)
            all_DE_ratio[act_prog_genes] = DEratio
            act_prog_mean = self.geneparams['gene_mean'] * all_DE_ratio
            for i, vec in enumerate(self.geneparams['act_prog_gene_mean']):
                vec[act_prog_idx] = act_prog_mean[i]

            for cell_type in self.act_prog_cell_types[act_prog_idx]:
                # For each cell type, let's go ahead and simulate the activity of this program if it is
                # relevant for this activity program at at all
                group_cells = self.cellparams.index[self.cellparams['cell_type'] == cell_type]

                # Determine which cells in this cell type group will have this specific activation program in it
                # We do this using a random Bernoulli distribution using a given expected fraction of cell types
                # that we expect to be activated
                has_prog = np.random.choice(
                    [True, False],
                    size=len(group_cells),
                    p=[
                        self.act_prog_cell_frac[act_prog_idx],
                        1 - self.act_prog_cell_frac[act_prog_idx],
                    ],
                )

                for vec in self.cellparams.loc[group_cells[has_prog], 'has_act_program']:
                    vec[act_prog_idx] = 1
                # And simulate usuage from a uniform distribution
                usages = np.random.uniform(
                    low=self.min_act_prog_usage[act_prog_idx],
                    high=self.max_act_prog_usage[act_prog_idx],
                    size=len(group_cells[has_prog]),
                )
                for i, vec in enumerate(self.cellparams.loc[group_cells[has_prog], 'act_program_usage']):
                    vec[act_prog_idx] = usages[i]



    def simulate_cell_types(self):
        '''Sample cell type identities from a categorical distribution'''
        cell_type_ids = np.random.choice(
            np.arange(1, self.n_cell_types + 1),
            size=self.init_ncells,
            p=self.cell_type_probs,
        )
        self.cell_types = np.unique(cell_type_ids)
        return cell_type_ids


    def sim_cell_type_DE(self):
        '''Sample differentially expressed genes and the DE factor for each
        cell-type'''
        cell_types = self.cellparams['cell_type'].unique()

        # Find out which genes are used for activity programs already to avoid differientating those
        if (self.act_prog_gene_sizes is not None) and np.sum(self.act_prog_gene_sizes) > 0:
            act_prog_genes = np.sum(
                np.array(list(map(list, self.geneparams['act_prog_gene'].values))),
                axis=-1,
            ) > 0
        else:
            act_prog_genes = np.array([False] * self.geneparams.shape[0])

        for cell_type in self.cell_types:
            isDE = np.array([False for _ in range(self.ngenes)])
            for gene_idx in range(self.ngenes):
                isDE[gene_idx] = np.random.choice(
                    [True, False],
                    p=[self.diffexpprob[cell_type - 1][gene_idx], 1 - self.diffexpprob[cell_type - 1][gene_idx]],
                )
            isDE[act_prog_genes] = False # Program genes shouldn't be differentially expressed between cell types
            DEratio = np.random.lognormal(
                mean=self.diffexploc,
                sigma=self.diffexpscale,
                size=isDE.sum(),
            )
            DEratio[DEratio < 1] = 1 / DEratio[DEratio<1]
            is_downregulated = np.random.choice(
                [True, False],
                size=len(DEratio),
                p=[self.diffexpdownprob, 1 - self.diffexpdownprob],
            )
            DEratio[is_downregulated] = 1 / DEratio[is_downregulated]
            all_DE_ratio = np.ones(self.ngenes)
            all_DE_ratio[isDE] = DEratio
            cell_type_mean = self.geneparams['gene_mean'] * all_DE_ratio

            self.geneparams[f'cell_type_{cell_type}_DE_ratio'] = all_DE_ratio
            self.geneparams[f'cell_type_{cell_type}_gene_mean'] = cell_type_mean
            self.geneparams[f'cell_type_{cell_type}_gene_selection'] = isDE
