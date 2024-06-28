from rogi import RoughnessIndex

import time
import copy
import random
from tqdm import tqdm
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    from botorch.models import SingleTaskGP, ModelListGP
    from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
    from botorch import fit_gpytorch_model
    from botorch.acquisition.monte_carlo import qExpectedImprovement
    from botorch.acquisition.monte_carlo import qUpperConfidenceBound
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
except:
    pass



warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_probs(init_weights, temperature=0.5):
    init_weights = torch.cat([torch.zeros(init_weights.size(0), 1).to(init_weights.device), init_weights], dim=-1)
    init_probs = torch.softmax(init_weights / temperature, dim=-1)
    return init_probs


def compute_ri(graph_reps, labels, init_weights, temperature=0.5, metric='euclidean'):
    assert len(labels.size()) == 2
    init_probs = get_probs(init_weights, temperature=temperature)
    graph_rep_ps = torch.matmul(graph_reps.transpose(0, 2), init_probs.transpose(0, 1)).transpose(0, 2)
    ri_ps = np.zeros(graph_rep_ps.size(0))
    for li in range(labels.size(1)):
        ri_ps += np.array([RoughnessIndex(Y=labels[:, li].cpu(),
                                          X=graph_rep_ps[i].cpu(),
                                          metric=metric,
                                          verbose=False).compute_index() for i in
                           range(graph_rep_ps.size(0))]) / labels.size(1)
    ri_ps = torch.Tensor(ri_ps).unsqueeze(-1).to(graph_reps.device)
    return ri_ps


def generate_data(graph_reps, labels, num_channels=3, num_inits=10, temperature=0.5, metric='euclidean'):
    # Generate weight via uniform distribution between (-1, 1):
    # init_weights = torch.rand(size=(num_inits, num_channels)).to(graph_reps.device)
    # init_weights = init_weights * 2 - 1
    init_weights = torch.normal(mean=0, std=1, size=(num_inits, num_channels)).to(graph_reps.device)
    ri_ps = -compute_ri(graph_reps, labels, init_weights, temperature=temperature, metric=metric)
    best_observed_value = ri_ps.max().item()

    return init_weights.double(), ri_ps.double(), best_observed_value


def get_next_points(init_x, init_y, best_init_y, bounds, n_points=1, n_restarts=500, n_samples=512):
    single_model = SingleTaskGP(init_x, init_y).to(init_x.device)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model).to(init_x.device)
    fit_gpytorch_model(mll)

    sampler = SobolQMCNormalSampler(sample_shape=256)
    acq_func = qExpectedImprovement(model=single_model, best_f=best_init_y, sampler=sampler)
    # acq_func = qUpperConfidenceBound(model=single_model, beta=5)

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=n_points,
        num_restarts=n_restarts,
        raw_samples=n_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return candidates


def optimize_prompt_weight_ri(graph_reps, labels, n_runs, n_inits, n_points, n_restarts, n_samples, temperature, metric,
                              skip_bo=False, verbose=False):
    start_time = time.time()
    n_channels = graph_reps.size(0)

    init_x, init_y, best_init_y = \
        generate_data(graph_reps, labels, num_channels=n_channels-1, num_inits=n_inits,
                      temperature=temperature, metric=metric)

    if verbose:
        print("Best init point:", -best_init_y)
        print("Best init prompt probs:", get_probs(init_x[torch.argmax(init_y)].unsqueeze(0),
                                                   temperature=temperature))

    if not skip_bo:
        best_init_y_prev = best_init_y
        bounds = torch.tensor([[-2.] * (n_channels), [2.] * (n_channels)]).to(graph_reps.device)
        if verbose:
            progress_bar = tqdm(range(n_runs))
        else:
            progress_bar = range(n_runs)

        for i in progress_bar:
            new_candidates = get_next_points(init_x, init_y, best_init_y, bounds, n_points, n_restarts, n_samples)
            new_results = -compute_ri(graph_reps, labels, new_candidates, temperature=temperature, metric=metric)

            init_x = torch.cat([init_x, new_candidates])
            init_y = torch.cat([init_y, new_results])
            best_init_y = init_y.max().item()

            if verbose:
                print(f"Iteration {i + 1:>3}/{n_runs} - Loss: {-new_results.sum().item():>4.3f}")
                # print("New prompt weights:")
                # print(get_probs(new_candidates, temperature=temperature))
                if best_init_y != best_init_y_prev:
                    print("Best new point:", -best_init_y)
                    print("Best new prompt probs:", get_probs(init_x[torch.argmax(init_y)].unsqueeze(0),
                                                              temperature=temperature))
                    best_init_y_prev = best_init_y

    best_weight = init_x[torch.argmax(init_y)]
    best_weight = torch.cat([torch.zeros(1).to(best_weight.device), best_weight], dim=-1)
    if verbose:
        print("Best prompt weights:", best_weight)
        print("Best prompt probs:", get_probs(init_x[torch.argmax(init_y)].unsqueeze(0), temperature=temperature))

    if verbose:
        print("Time elapsed: {} sec".format(np.round(time.time() - start_time, 2)))
    return best_weight


