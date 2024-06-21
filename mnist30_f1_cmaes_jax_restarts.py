import os
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
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import jax
import os
import pickle
from evosax import CMA_ES, IPOP_CMA_ES

if __name__ == "main":
    os.makedirs("./out", exist_ok=True)

    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = DataLoader(testset, batch_size=10000, shuffle=False)

    block_size = 100
    # Define model
    from src.models import MNIST30K

    model = MNIST30K()
    # model = vgg16()
    # model.classifier = nn.Sequential(*model.classifier, nn.Linear(1000, 10))
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
    x0 = np.random.uniform(low=-5, high=5, size=(bD))
    init_params_blocked = blocker(init_params, codebook)
    x0 = init_params_blocked.copy()
    # init_pop = np.random.normal(loc=init_params, scale=0.1, size=(NP, bD))
    batch_size = 128

    problem = GFOProblem(
        n_var=bD,
        model=model,
        dataset=trainset,
        test_loader=test_loader,
        train_loader=None,
        set_model_state=set_model_state,
        batch_size=batch_size,
        device=device,
        criterion="f1",
        block=True,
        codebook=codebook,
        orig_dims=D,
    )
    out = {"F": []}
    problem._evaluate([x0, init_params, init_params_blocked], out=out)
    print(out)
    csv_path = (
        f"out/MNIST30K_block_bs{block_size}_gfo_100Kfe_f1_cmaes_jax_restart5_hist.csv"
    )
    plt_path = (
        f"out/MNIST30K_block_bs{block_size}_gfo_100Kfe_f1_cmaes_jax_restart5_plt.pdf"
    )
    df = pd.DataFrame(
        {
            "n_step": [0],
            "n_eval": [1],
            "f_best": [out["F"][0]],
            "f_avg": [out["F"][0]],
            "f_std": [0],
            "test_f1_best": problem.test_func(x0),
        }
    )
    df.to_csv(csv_path, index=False)

    restarts = 5
    # best_x0 = x0
    best_x0 = np.zeros(bD)
    best_F = df["f_best"][0]
    best_state = None
    curr_iter = 0
    maxFE = 100000

    problem = GFOProblem(
        n_var=bD,
        model=model,
        dataset=trainset,
        test_loader=test_loader,
        train_loader=None,
        set_model_state=set_model_state,
        batch_size=batch_size,
        device=device,
        criterion="f1",
        block=True,
        codebook=codebook,
        orig_dims=D,
    )

    rng = jax.random.PRNGKey(1)

    optimizer = IPOP_CMA_ES(
        popsize=4 + int(3 * np.log(bD)), num_dims=bD, sigma_init=1.0
    )  # , bounds=np.array([[-1, 1]]*bD))
    NP = optimizer.strategy.popsize
    es_params = optimizer.default_params.replace(
        strategy_params=optimizer.default_params.strategy_params.replace(
            clip_min=-5, clip_max=5
        ),
        restart_params=optimizer.default_params.restart_params.replace(
            min_fitness_spread=1, min_num_gens=500
        ),
    )
    if best_state is None:
        state = optimizer.initialize(rng, es_params)
    else:
        state = best_state

    FE = 0
    iters = 0
    res_counter = 0
    callback = SOCallback(
        k_steps=100,
        csv_path=csv_path,
        plt_path=plt_path,
        start_eval=FE,
        start_iter=curr_iter,
        problem=problem,
    )
    print(
        f"n_steps \t| n_evals \t| best F \t| pop F_min \t| pop F_mean \t| pop F_std \t| sigma"
    )
    while FE < maxFE:
        rng, rng_gen, rng_eval = jax.random.split(rng, 3)
        if state.restart_state.restart_next:
            print("Restart optimizer #:", res_counter + 1)
            # Save the best solution model parameters state
            best_X = best_x0.copy()
            if len(best_X) != D:
                best_X = problem.unblocker(best_X)
            set_model_state(model, best_X)
            torch.save(
                model.state_dict(),
                f"out/MNIST30K_block_bs{block_size}_gfo_100Kfe_f1_cmaes_jax_restart5_model_{res_counter}.pth",
            )
            res_counter += 1
            batch_size *= 2
            curr_iter += iters

            # state.restart_state.replace(min_num_gens=maxFE//optimizer.base_strategy.popsize)

            problem = GFOProblem(
                n_var=bD,
                model=model,
                dataset=trainset,
                test_loader=test_loader,
                train_loader=None,
                set_model_state=set_model_state,
                batch_size=batch_size,
                device=device,
                criterion="f1",
                block=True,
                codebook=codebook,
                orig_dims=D,
            )

            callback = SOCallback(
                k_steps=100,
                csv_path=csv_path,
                plt_path=plt_path,
                start_eval=FE,
                start_iter=curr_iter,
                problem=problem,
            )

            NP *= 2

        pop_F = np.zeros(NP)
        pop_X = np.zeros((NP, bD))
        pop_X, state = optimizer.ask(rng_gen, state, es_params)
        for ip in range(NP):
            f = problem.scipy_fitness_func(pop_X[ip])
            pop_F[ip] = f

        state = optimizer.tell(pop_X, pop_F, state, es_params)

        argmin = pop_F.argmin()
        min_F = pop_F[argmin]
        if min_F < best_F:
            best_F = min_F
            best_x0 = pop_X[argmin]

        FE += NP
        iters += 1
        callback.general_caller(
            niter=iters, neval=FE, opt_X=best_x0, opt_F=best_F, pop_F=pop_F
        )
        print(
            f"{iters} \t| {FE} \t| {best_F:.6f} \t| {pop_F.min():.6f} \t| {pop_F.mean():.6f} \t| {pop_F.std():.6f} \t| {state.strategy_state.sigma:.6f}"
        )

    print("Best solution found: \nX = %s\nF = %s" % (best_x0, best_F))
