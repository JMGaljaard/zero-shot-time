import typing as tp

import jax
import numpy as np
import optuna.study
import torch
import transformers.models.gpt2.modeling_gpt2
from optuna import Study

from zero_shot_time.data import pre_processing
from optuna.samplers import BruteForceSampler
from optuna.trial import Trial

from zero_shot_time.data.scaler import Scaler
from zero_shot_time.generation.nll import calculate_negative_log_likelihood


def process_sets(
        param_sets: tp.List[tp.Tuple[tp.List[np.array], tp.List[np.array]]],
        precision: int = 3,
        tokenizer: transformers.PreTrainedTokenizerFast = None,
        drop_test_comma: bool = True,
        seperator=None,
        form: str = ' {}',
        **scaler_kwargs
) -> (tp.List[torch.LongTensor], tp.List[torch.LongTensor], tp.List[Scaler]):
    """Helper function to pre-process a list of train and test sets, using the training sets to re-scale training and
    testing data.
    Args:
        param_sets (tp.L): List of lists containing pairs of training and testing sets to pre-process individually.
        precision (): Numerical precision to use during encoding of the tokenized representations after scaling.
        tokenizer (): Tokenizer (corresponding to a model) to use to perform analysis on.
        drop_test_comma (bool, default True):
            Optional configuration to keep last comma of test input_ids. Defaults to drop the last comma/seperator.
        **scaler_kwargs (): Keyword arguments (hyper-parameters) to pass to the scaler used during pre-processing.

    Returns:
        List of training input encodings to token_ids.
        List of targets/future input encodings to token_ids.
        List of scaler scores
    """
    train_sets, test_sets = [], []
    hyper_scalers = []
    for train_set, test_set in param_sets:
        predictions = len(test_set) - len(train_set)
        # 1. Pre-process training and compute scaling object (prevent knowledge leakage through scaling
        scaler, process_values_train, input_ids_train = pre_processing.convert_timeseries_to_fixed_precision(
            None, tokenizer, values=train_set, precision=precision, seperator=seperator, form=form, **scaler_kwargs
        )
        # 2. Pre-process test and re-use scaling so tha
        _, process_values_test, input_ids_test = pre_processing.convert_timeseries_to_fixed_precision(
            None, tokenizer, values=train_set + test_set, pre_processor=scaler, precision=precision, seperator=seperator, form=form, **scaler_kwargs
        )

        if drop_test_comma:
            # Remove the last comma seperated value, as we are not going to predict futher than that
            input_ids_test = input_ids_test[:-1]

        train_sets.append(input_ids_train.to('cuda'))
        test_sets.append(input_ids_test.to('cuda'))
        hyper_scalers.append(scaler)

    return train_sets, test_sets, hyper_scalers

def curried_hyper_opt(
    study: Study,
    train_eval_sets: tp.List[tp.Tuple[np.array, np.array]],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    allowable_tokens_mask: torch.BoolTensor,
    seperator_token_id: int,
    seperator: str,
    form: str,
    padding_token_id: int,
    alpha: tp.List[float] = None,
    beta: tp.List[float] = None,
    precision: tp.List[int] = None,
    tau: tp.List[float] = None,
    offset: int = 0,
):
    """Curried applicatble function to perform hyper-parameter tuning with grid search. Note that not all parameters are
    'required', but are presently provided with debugging and extension in mind.

    Args:
        train_sets (List[(np.array, np.array)]): List of Tuples consisting fo matching training and testing sets.
        model (): Language models to perform learning with.
        tokenizer (): Tokenizer corresponding to the models.
        allowable_tokens_mask (): Token mask corresponding to the set of allowable numerical tokens, this is required to
            efficiently compute the negative log-likely hood of token encodings.
        seperator_token_id ():  Seperator ID, used for sanity checking input_token_ids.
        padding_token_id (): (optional) padding token id for intra-value paddding of numerical representations.
        alpha (List[float]): (categorical) list of hyper-parmeter values to use during grid-search for alpha (scaler).
        beta (List[float]): (categorical) list of hyper-parmeter values to use during grid-search for beta (scaler).
        precision (): (categorical) list of hyper-parmeter values to use during grid-search for encoding (precision).
        tau (): (categorical) list of hyper-parmeter values to use during grid-search for encoding.
        offset (int): Number of 'offset' tokens that are added by default tokenization as expected by the model.
            Some newer models require a set of `start prompt' tokens that need to be accounted for.
    Returns:

    """
    # Store hyper-parameters locally
    local_study = study
    alpha_grid = alpha
    beta_grid = beta
    precision_grid = precision
    tau_grid = tau
    local_offset = offset
    # TODO: Check if precision requires actual evaluation, as in general longer sequences tend to have higher logits.
    def llm_hyper_opt(
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
        # 1. Select hyper-parameters to evaluate on.
        # Retrieve random selection to perform evaluation on.
        tau = trial.suggest_categorical(name='tau',
                                        choices=tau_grid)

        alpha = trial.suggest_categorical('alpha',
                                          choices=alpha_grid)

        beta = trial.suggest_categorical('beta',
                                         choices=beta_grid)
        precision = trial.suggest_categorical('precision',
                                              choices=precision_grid)

        # 1. Create tokenized representation

        # 2. Pre-process and tokenize data
        train_sets_tokens, test_sets_tokens, scalers = process_sets(
            train_eval_sets,
            precision=precision,                                # Precision (decimal) to continue with
            tokenizer=tokenizer,                                #
            quantile=alpha,                                        # Scaler alpha
            beta=beta,                                          # Scaler beta
            seperator=seperator,
            form=form,
        )

        # 3. Calculate NLL statistic on pre-processed data.
        nll_aggregate = 0
        for (train_value, test_value), train_set, test_set in zip(train_eval_sets, train_sets_tokens, test_sets_tokens):
            # Retrieve grad (scalar) from learned scaler
            dy_dx = jax.vmap(jax.grad(scalers[0].transform))(
                np.array(train_value)
            ).mean().item()
            logits, nll_ = calculate_negative_log_likelihood(
                model=model,                                    #
                input_token_ids=train_set,                      # Provide input in token_id representation
                target_token_ids=test_set,                      # Provide targets in token_id representation
                separator_token_id=seperator_token_id,          #
                padding_token_id=padding_token_id,              #
                dy_dx=dy_dx,                                    # Conversion from scaled back to 'ordinary' values.
                token_mask=allowable_tokens_mask,               #
                precision=precision,
                base=10,
                temperature=tau,
                pre_computed_logits=None,                      # Possibly re-use some results during hyper-parameter search,
                offset=local_offset,
                prediction_len=len(test_value) - len(train_value)    # Prediction length
            )
            nll_aggregate += nll_
        # Return the average nll score to the caller for hyper-parameter optimization.
        return nll_aggregate / len(train_eval_sets)

    # Return curried function to be called with local environment.
    return llm_hyper_opt



def perform_hyper_parameter_tuning(
        dataset_name: str,
        model_name: str,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerFast,
        data_sets: tp.List[tp.List[tp.Tuple[np.array, np.array]]],
        allowable_token_mask: torch.BoolTensor,
        seperator_token_id: int,
        seperator: str,
        form: str,
        padding_token_id: int,
        search_space: dict[str, tp.List[tp.Any]],
        offset: int = 0,
) -> Study:
    """

    Args:
        dataset_name: Dataset name (for creating a logging file
        model_name:
        model:
        tokenizer:
        data_sets:
        allowable_token_mask:
        seperator_token_id:
        seperator:
        form:
        padding_token_id:
        search_space:
        offset (int): Number of 'offset' tokens that need to be discarded from history.
    Returns:

    """
    name = f"{dataset_name}_{model_name}"
    storage = f"sqlite:///{name.split('/')[-1]}.db"
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        sampler=BruteForceSampler(seed=42),
        load_if_exists=True
    )
    max_trails = np.prod([len(v) for _, v in search_space.items()])

    partial_applied_search = curried_hyper_opt(
        study=study,
        train_eval_sets=data_sets,
        model=model,
        tokenizer=tokenizer,
        seperator=seperator,
        allowable_tokens_mask=allowable_token_mask,
        seperator_token_id=seperator_token_id,
        form=form,
        padding_token_id=padding_token_id,
        alpha=search_space['alpha'],
        beta=search_space['beta'],
        precision=search_space['precision'],
        tau=search_space['tau'],
        offset=offset,
    )
    # Because we peform brute-force search, we don't specify number of trails.
    study.optimize(partial_applied_search, n_trials=max_trails)
    return study