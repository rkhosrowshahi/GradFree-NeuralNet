import os
from time import time

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from src.gfo import (
    GFOProblem,
    SOCallback,
    blocker,
    build_rand_blocks,
    get_model_params,
    set_model_state,
)

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models import MLP

import numpy as np
import pandas as pd
import jax
import os
import pickle
from evosax import CMA_ES

from ucimlrepo import fetch_ucirepo


if __name__ == "__main__":
    os.makedirs("./out", exist_ok=True)

    # Load the iris dataset
    uci_dataset = fetch_ucirepo(id=225)

    X = uci_dataset.data.features
    X["Gender"] = X["Gender"].apply(lambda x: 1 if x == "Male" else 0)
    y = uci_dataset.data.targets.to_numpy().flatten() - 1

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Create TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split dataset into train (80%) and test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training and testing
    batch_size = train_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    num_classes = 2

    block_size = 5

    model = MLP(
        in_size=X.shape[1],
        hidden_size=X.shape[1] * X.shape[1] + 1,
        out_size=num_classes,
    )
    print(model)
    problem_name = "liver-MLP-2N"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    init_params = get_model_params(model)
    D = len(init_params)
    print(f"Original dims: {D} D")

    codebook = {}
    if os.path.exists(f"out/codebook_D{D}_blocksize{block_size}.pkl"):
        with open(f"out/codebook_D{D}_blocksize{block_size}.pkl", "rb") as f:
            codebook = pickle.load(f)
    else:
        codebook = build_rand_blocks(D, block_size=block_size)

        with open(f"out/codebook_D{D}_blocksize{block_size}.pkl", "wb") as f:
            pickle.dump(codebook, f)

    bD = len(codebook)
    print(f"Blocked dims: {bD} D")
    init_params_blocked = blocker(init_params, codebook)
    x0 = np.zeros(bD)

    problem = GFOProblem(
        n_var=bD,
        model=model,
        dataset=None,
        test_loader=test_loader,
        train_loader=train_loader,
        set_model_state=set_model_state,
        batch_size=batch_size,
        num_classes=num_classes,
        device=device,
        criterion="f1",
        block=True,
        codebook=codebook,
        orig_dims=D,
    )
    out = {"F": []}
    problem._evaluate([x0, init_params, init_params_blocked], out=out)
    print(out)
    csv_path = f"out/{problem_name}_block_bs{block_size}_gfo_100Kfe_top1_hist.csv"
    plt_path = f"out/{problem_name}_block_bs{block_size}_gfo_100Kfe_top1_plt.pdf"
    model_path = f"out/{problem_name}_block_bs{block_size}_gfo_100Kfe_top1_model"
    test_fs = problem.test_func(init_params_blocked)
    test_f1, test_top1 = test_fs["f1"], test_fs["top1"]
    df = pd.DataFrame(
        {
            "n_step": [0],
            "n_eval": [1],
            "f_best": [out["F"][2]],
            "f_avg": [0],
            "f_std": [0],
            "test_f1_best": [test_f1],
            "test_top1_best": [test_top1],
        }
    )
    df.to_csv(csv_path, index=False)

    restarts = 5
    # best_x0 = x0
    best_x0 = np.zeros(bD)
    best_F = df["f_best"][0]
    best_state = None
    curr_iter = 0
    maxFE = 25000
    resFE = 200000

    rng = jax.random.PRNGKey(1)

    optimizer = CMA_ES(popsize=100, num_dims=bD, sigma_init=0.5, elite_ratio=0.5)
    NP = optimizer.popsize
    es_params = optimizer.default_params
    if best_state is None:
        state = optimizer.initialize(rng, es_params)
        # state.replace(mean=best_x0)
        # state.replace(clip_min=-5, clip_max=5)
    else:
        state = best_state

    FE = 0
    iters = 0
    res_counter = 0
    callback = SOCallback(
        k_steps=10,
        csv_path=csv_path,
        plt_path=plt_path,
        start_eval=FE,
        start_iter=curr_iter,
        problem=problem,
    )
    print(
        f"n_steps\t,n_evals,best F\t,pop F_best\t,pop F_mean\t,pop F_std\t,pop X_low\t,pop X_high\t,sigma\t\t,optimization time(s)\t,evaluation time(s)"
    )
    while FE < maxFE:
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)

        pop_F = np.zeros(NP)
        pop_X = np.zeros((NP, bD))
        opt_t1 = time()
        pop_X, state = optimizer.ask(rng_gen, state, es_params)
        opt_t2 = time()
        eval_t1 = time()
        for ip in range(NP):
            f = problem.general_fitness_func(pop_X[ip])
            pop_F[ip] = f
        eval_t2 = time()

        opt_t3 = time()
        state = optimizer.tell(pop_X, pop_F, state, es_params)
        opt_t4 = time()

        argmin = pop_F.argmin()
        min_F = pop_F[argmin]
        if min_F <= best_F:
            best_F = min_F
            best_x0 = pop_X[argmin]

            best_X = best_x0.copy()
            if len(best_X) != D:
                best_X = problem.unblocker(best_X)
            set_model_state(model, best_X)
            # torch.save(
            #     model.state_dict(), f"{model_path}_{iters}iteration_{FE}fe_best.pth"
            # )
            torch.save(model.state_dict(), f"{model_path}_best.pth")

        FE += NP
        iters += 1

        callback.general_caller(
            niter=iters, neval=FE, opt_X=best_x0, opt_F=best_F, pop_F=pop_F
        )
        if iters % 10 == 0:
            print(
                f"{iters}\t,{FE}\t,{best_F:.6f}\t,{pop_F.min():.6f}\t,{pop_F.mean():.6f}\t,{pop_F.std():.6f}\t,{pop_X[argmin].min():.6f}\t,{pop_X[argmin].max():.6f}\t,{state.sigma:.6f}\t,{((opt_t2-opt_t1) + (opt_t4-opt_t3)):.6f}\t,{(eval_t2-eval_t1):.6f}"
            )

    callback.general_caller(
        niter=iters, neval=FE, opt_X=best_x0, opt_F=best_F, pop_F=pop_F, last_iter=True
    )

    print("Best solution found: \nX = %s\nF = %s" % (best_x0, best_F))
    # Save the best solution model parameters state
    best_X = best_x0.copy()
    if len(best_X) != D:
        best_X = problem.unblocker(best_X)
    set_model_state(model, best_X)
    torch.save(model.state_dict(), f"{model_path}_last.pth")
