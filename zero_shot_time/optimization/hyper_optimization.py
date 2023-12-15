import typing as tp

import numpy as np
import optuna.study
import transformers.models.gpt2.modeling_gpt2

from zero_shot_time.data.scaler import Scaler
from optuna.samplers import GridSampler, BruteForceSampler
from optuna.study import Study
from optuna.trial import Trial

def curried_hyper_opt(
    scalers: tp.List[Scaler],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    alpha: tp.List[float] = None,
    beta: tp.List[float] = None,
    precision: tp.List[int] = None,
    tau: tp.List[float] = None,
):
    alpha_grid = alpha
    beta_grid = beta
    precision_grid = precision
    tau_grid = tau
    def llm_hyper_opt(
            train_sets: tp.List[np.array],
            test_sets: tp.List[np.array],
            trial: Trial
    ) -> float:

        """Simple function to perform hyper-opt search using training data split into training and validation (test) sets,
        to perform hyper-parameter tuning. This is performed through a grid-search as described in the original paper.

        The object that is optimized is the average NLL score of the predictions, as to maximize the predictions of the
        outputs. It is assumed that the train and test sets are sanitized, etc. by the caller.

        Args:
            train_sets ():
            test_sets ():
            scalers ():
            model ():
            tokenizer ():
            hyper_params ():

        Returns:
            Negative log-likely hood, provided with a samples generated data.

        """
        # Retrieve random selection to perform evaluation on.
        tau = trial.suggest_categorical(name='tau',
                                        choices=tau_grid)
        alpha = trial.suggest_categorical('alpha',
                                          choices=alpha_grid)

        beta = trial.suggest_categorical('beta',
                                         choices=beta_grid)
        precision = trial.suggest_categorical('precision',
                                              choices=beta_grid)
        nll_aggregate = 0
        for train_set, test_set in zip(train_sets, test_sets):

            # 1. Create tokenized representation

            # 2. Forward with the model

            # 3. Evaluate log_likely_hood

            nll = ...

            nll_aggregate += nll_aggregate

        return nll_aggregate



if __name__ == '__main__':
    study = optuna.study.create_study(sample=BruteForceSampler(), direction='minimize')
    study.