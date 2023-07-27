import metrics
import numpy as np
import scipy
import logging
import sklearn
import training.utils as utils

def evaluate_concept_representations(
    end_results,
    experiment_config,
    test_concept_scores,
    c_test=None,
    y_test=None,
    old_results=None,
    load_from_cache=True,
    prefix="",
):
    if (c_test is not None) and (
        (experiment_config.get("n_supervised_concepts", 0) != 0) and
        (len(experiment_config.get('supervised_concept_idxs', [])) > 0)
    ):
        # Then compute the mean concept predictive accuracy
        supervised_concept_idxs = experiment_config['supervised_concept_idxs']
        avg = 0.0
        for learnt_concept_idx, real_concept_idx in enumerate(
            supervised_concept_idxs
        ):
            # And select just the labels that are in fact being used
            avg += sklearn.metrics.roc_auc_score(
                c_test[:, real_concept_idx],
                test_concept_scores[:, learnt_concept_idx],
            )
        end_results['avg_concept_auc'] = avg / len(supervised_concept_idxs)
        logging.debug(
            prefix +
            f"\t\tMean Concept AUC is {end_results['avg_concept_auc']*100:.2f}%"
        )


    if c_test is None:
        # Then nothing else to do in here
        return
    if not experiment_config.get('continuous_concepts', False):
        # Compute the CAS score
        logging.debug(prefix + "\t\tComputing CAS...")
        end_results['cas'], end_results['cas_task'], end_results['best_alignment'] = utils.posible_load(
            key=['cas', 'cas_task', 'best_alignment'],
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.embedding_homogeneity(
                c_vec=test_concept_scores,
                c_test=c_test,
                y_test=y_test,
                step=experiment_config.get('cas_step', 2),
            ),
        )
        logging.debug(
            prefix + f"\t\t\tDone with CAS = {end_results['cas'] * 100:.2f}%"
        )

        # Now compute MIG
        logging.debug(prefix + "\t\tComputing MIG...")
        end_results['mig'] = utils.posible_load(
            key='mig',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.MIG(
                Z_true=c_test,
                Z_learned=test_concept_scores,
                bins=experiment_config.get('num_bins', 10)
            ),
        )
        logging.debug(
            prefix + f"\t\t\tDone with MIG = {end_results['mig'] * 100:.2f}%"
        )

        # Now compute SAP
        logging.debug(prefix + "\t\tComputing SAP...")
        end_results['sap'] = utils.posible_load(
            key='sap',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.SAP(
                V=c_test,
                Z=test_concept_scores,
            ),
        )
        logging.debug(
            prefix + f"\t\t\tDone with SAP = {end_results['sap'] * 100:.2f}%"
        )

        # And go for FactorVAE
        logging.debug(prefix + "\t\tComputing FactorVAE Scores...")
        end_results['factor_vae'] = utils.posible_load(
            key='factor_vae',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.FactorVAE(
                ground_truth_X=test_concept_scores,
                ground_truth_Z=c_test,
                representation_function=lambda x: x,
                random_state=np.random.RandomState(0),
                batch_size=experiment_config.get('batch_size', 64),
                num_train=int(test_concept_scores.shape[0] * 0.7),
                num_eval=int(test_concept_scores.shape[0] * 0.3),
                num_variance_estimate=int(test_concept_scores.shape[0] * 0.3),
            ),
        )
        logging.debug(
            prefix + f"\t\t\tDone with FactorVAE = {end_results['factor_vae'] * 100:.2f}%"
        )

        # Then DCI
        logging.debug(prefix + "\t\tComputing DCI Scores...")
        end_results['dci_disentanglement'], end_results['dci_completeness'], end_results['dci_informativeness'] = utils.posible_load(
            key=['dci_disentanglement', 'dci_completeness', 'dci_informativeness'],
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.DCI(
                gen_factors=c_test,
                latents=test_concept_scores,
            ),
        )
        logging.debug(
            prefix +
            f"\t\t\tDone DCI disentanglement = "
            f"{end_results['dci_disentanglement']*100:.2f}%"
        )
        logging.debug(
            prefix +
            f"\t\t\tDone DCI completeness = "
            f"{end_results['dci_completeness']*100:.2f}%"
        )
        logging.debug(
            prefix +
            f"\t\t\tDone DCI informativeness = "
            f"{end_results['dci_informativeness']*100:.2f}%"
        )

        # R4 Scores
        logging.debug(prefix + "\t\tComputing R4 Score...")
        end_results['r4'] = utils.posible_load(
            key='r4',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.R4_scores(
                V=c_test,
                Z=test_concept_scores,
            ),
        )
        logging.debug(
            prefix + f"\t\t\tDone with R4 = {end_results['r4'] * 100:.2f}%"
        )

    # Compute correlation between bottleneck entries and ground truch concepts
    logging.debug(prefix + "\t\tConcept correlation matrix...")
    end_results['concept_corr_mat'] = utils.posible_load(
        key='concept_corr_mat',
        old_results=old_results,
        load_from_cache=load_from_cache,
        run_fn=lambda: metrics.correlation_alignment(
            scores=test_concept_scores,
            c_test=c_test,
        ),
    )
    logging.debug(prefix + f"\t\t\tDone")
