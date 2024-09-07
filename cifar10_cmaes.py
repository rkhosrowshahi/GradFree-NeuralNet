import argparse
from datetime import datetime
import os
from time import time
from src.gfo.utils import get_network
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
import torchsummary

import numpy as np
import pandas as pd
import jax
import os
import pickle
from evosax import CMA_ES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, required=True, help="network type")
    parser.add_argument(
        "--gpu", type=str, default="cuda:0", help="use cuda:[number1], cuda:[number2]"
    )
    parser.add_argument("--b", type=int, default=128, help="batch size for dataloader")
    parser.add_argument(
        "--bs", type=int, default=10, help="block size for fixed codebook blocking"
    )
    parser.add_argument(
        "--maxfe", type=int, default=1000000, help="max function evaluation"
    )
    parser.add_argument("--np", type=int, default=None, help="pop size")
    parser.add_argument("--sigma", type=float, default=0.1, help="initial sigma")
    parser.add_argument("--criterion", type=str, default="top1", help="use top1 or f1")
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="imagenet",
    #     help="use cifar10, cifar100 or imagenet",
    # )
    # parser.add_argument(
    #     "--seed", type=int, default=1, help="seed value for random values"
    # )
    # parser.add_argument(
    #     "--lb", type=int, default=2, help="lower bound in population initilization"
    # )
    # parser.add_argument(
    #     "--ub", type=int, default=1000, help="upper bound in population initilization"
    # )
    # parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    device = torch.device(args.gpu)
    maxFE = args.maxfe
    block_size = args.bs
    b = args.b
    popsize = args.np
    sigma_init = args.sigma

    # Load MNIST
    num_classes = 10
    transform_train = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(trainset, batch_size=b, shuffle=True)
    test_loader = DataLoader(testset, batch_size=b, shuffle=False)

    model = get_network(args.net, num_classes=num_classes)
    network_name = args.net
    dataset_name = "cifar10"
    model.to(device)
    print(torchsummary.summary(model, (3, 32, 32)))

    init_params = get_model_params(model)
    D = len(init_params)
    print(f"Original dims: {D} D")

    codebook_path = os.path.join(
        "codebooks",
        dataset_name,
        network_name,
    )
    if not os.path.exists(codebook_path):
        os.makedirs(codebook_path)
    codebook_path = os.path.join(codebook_path, "{net}-bs{bs}.pkl")

    codebook = {}
    codebook_path_file = codebook_path.format(net=network_name, bs=block_size)
    if os.path.exists(codebook_path_file):
        with open(codebook_path_file, "rb") as f:
            codebook = pickle.load(f)
    else:
        codebook = build_rand_blocks(D, block_size=block_size)
        with open(codebook_path_file, "wb") as f:
            pickle.dump(codebook, f)

    bD = len(codebook)
    print(f"Blocked dims: {bD} D")
    init_params_blocked = blocker(init_params, codebook)
    x0 = np.zeros(bD)

    problem = GFOProblem(
        n_var=bD,
        model=model,
        dataset=trainset,
        test_loader=test_loader,
        train_loader=None,
        set_model_state=set_model_state,
        batch_size=b,
        device=device,
        criterion="top1",
        block=True,
        codebook=codebook,
        orig_dims=D,
    )

    out = {"F": []}
    problem._evaluate([init_params_blocked, init_params, x0], out=out)
    print(out)

    DATE_FORMAT = "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
    checkpoint_path = os.path.join(
        "out",
        dataset_name,
        network_name,
        f"bs{block_size}-D{bD}",
        datetime.now().strftime(DATE_FORMAT),
    )
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    csv_path_format = os.path.join(checkpoint_path, "{dataset}-{net}-{obj}_hist.csv")
    csv_path = csv_path_format.format(
        dataset=dataset_name, net=network_name, obj=problem.criterion
    )
    plt_path_format = os.path.join(checkpoint_path, "{dataset}-{net}-{obj}_plot.pdf")
    plt_path = plt_path_format.format(
        dataset=dataset_name, net=network_name, obj=problem.criterion
    )
    model_path_format = os.path.join(
        checkpoint_path, "{dataset}-{net}-{obj}-{type}_model.pth"
    )

    test_fs = problem.test_func(init_params_blocked)
    test_f1, test_top1 = test_fs["f1"], test_fs["top1"]
    df = pd.DataFrame(
        {
            "n_step": [0],
            "n_eval": [1],
            "f_best": [out["F"][0]],
            "f_avg": [0],
            "f_std": [0],
            "test_f1_best": [test_f1],
            "test_top1_best": [test_top1],
        }
    )
    df.to_csv(csv_path, index=False)

    best_x0 = init_params_blocked
    best_F = df["f_best"][0]
    best_state = None
    curr_iter = 0

    rng = jax.random.PRNGKey(1)

    if popsize == None:
        popsize = 4 + int(3 * np.log(bD))
    optimizer = CMA_ES(popsize=popsize, num_dims=bD, sigma_init=sigma_init)
    NP = optimizer.popsize
    es_params = optimizer.default_params
    if best_state is None:
        state = optimizer.initialize(rng, es_params)
        state.replace(mean=best_x0)
    else:
        state = best_state

    FE = 0
    iters = 0

    callback = SOCallback(
        k_steps=100,
        csv_path=csv_path,
        plt_path=plt_path,
        start_eval=FE,
        start_iter=iters,
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
            model_path = model_path_format.format(
                dataset=dataset_name,
                net=network_name,
                obj=problem.criterion,
                type="best",
            )
            torch.save(model.state_dict(), model_path)

        FE += NP
        iters += 1

        callback.general_caller(
            niter=iters, neval=FE, opt_X=best_x0, opt_F=best_F, pop_F=pop_F
        )
        if iters % 1 == 0:
            print(
                f"{iters}\t,{FE}\t,{best_F:.6f}\t,{pop_F.min():.6f}\t,{pop_F.mean():.6f}\t,{pop_F.std():.6f}\t,{pop_X[argmin].min():.6f}\t,{pop_X[argmin].max():.6f}\t,{state.sigma:.6f}\t,{((opt_t2-opt_t1) + (opt_t4-opt_t3)):.6f}\t,{(eval_t2-eval_t1):.6f}"
            )

    print("Best solution found: \nX = %s\nF = %s" % (best_x0, best_F))
    # Save the best solution model parameters state
    best_X = best_x0.copy()
    if len(best_X) != D:
        best_X = problem.unblocker(best_X)
    set_model_state(model, best_X)
    model_path = model_path_format.format(
        dataset=dataset_name, net=network_name, obj=problem.criterion, type="last"
    )
    torch.save(model.state_dict(), model_path)
