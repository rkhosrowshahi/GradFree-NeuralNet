import itertools
import sys
from time import sleep
import time
import torch
from torch.utils.data import DataLoader, Subset

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    top_k_accuracy_score,
)

from torcheval.metrics.functional import multiclass_f1_score, topk_multilabel_accuracy
import matplotlib.pyplot as plt


from pymoo.core.problem import Problem
from pymoo.core.callback import Callback

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


class GFOProblem(Problem):
    def __init__(
        self,
        n_var=None,
        model=None,
        dataset=None,
        batch_size=1024,
        num_classes=10,
        block=False,
        codebook=None,
        orig_dims=None,
        set_model_state=None,
        device=None,
        criterion=None,
        test_loader=None,
        train_loader=None,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=0,
            xl=-5.0,
            xu=5.0,
            vtype=float,
        )
        self.model = model
        self.batch_size = batch_size
        self.block = block  # Enable / disable block
        self.dataset = dataset
        self.num_classes = num_classes
        self.test_loader = test_loader
        self.set_model_state = set_model_state
        self.device = device
        self.codebook = codebook
        self.orig_dims = orig_dims
        self.criterion = criterion
        if criterion is None:
            self.fitness_func = self.f1score_func
        elif criterion == "crossentropy":
            self.fitness_func = self.crossentropy_func
        elif criterion == "f1":
            self.fitness_func = self.f1score_func
        elif criterion == "top1":
            self.fitness_func = self.top1_func

        if train_loader is None:
            self.data_loader = self.data_sampler()
        else:
            self.data_loader = train_loader

        self.fitness = None

    def data_sampler(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=2,
            pin_memory=True,
        )

    # def unblocker(self, blocked_params):

    #     unblocked_params = np.zeros(self.orig_dims)
    #     # start_time = time.time()
    #     for block_idx, indices in self.codebook.items():
    #         # st_in = time.time()
    #         unblocked_params[indices] = np.full(len(indices), blocked_params[block_idx])
    #         # tot_in = time.time() - st_in
    #         # print(tot_in)

    #     # end_time = time.time() - start_time

    #     # print(end_time)
    #     return unblocked_params

    def update_unblocked_params(self, block_idx, indices, blocked_params):
        self.unblocked_params[indices] = np.full(
            len(indices), blocked_params[block_idx]
        )

    def unblocker(self, blocked_params):
        self.unblocked_params = np.zeros(self.orig_dims)
        with ThreadPoolExecutor() as executor:
            futures = []
            for block_idx, indices in self.codebook.items():
                futures.append(
                    executor.submit(
                        self.update_unblocked_params, block_idx, indices, blocked_params
                    )
                )

            for future in futures:
                future.result()  # Wait for all futures to complete

        return self.unblocked_params

    # def assign_values(self, args):
    #     b = np.zeros(self.orig_dims)
    #     a, indices = args[0], args[1]

    #     b[indices] = a
    #     # b[0] = indices[0]
    #     return b

    # def unblocker2(self, blocked_params):
    #     flattened_indices = []
    #     for i, indices in self.codebook.items():
    #         flattened_indices.extend(indices)

    #     unblocked_params = np.zeros(self.orig_dims)

    #     st_in = time.time()
    #     with mp.Pool(processes=4) as pool:
    #         result = pool.imap(
    #             self.assign_values,
    #             zip(blocked_params, self.codebook.values()),
    #         )
    #     tot_in = time.time() - st_in
    #     print(tot_in)

    #     unblocked_params = np.sum(result, 0)

    #     return unblocked_params

    def crossentropy_func(self, model, data_loader, device):
        model.eval()
        fitness = 0

        data, target = next(iter(data_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        fitness += torch.nn.functional.cross_entropy(output, target).item()

        return fitness

    def f1score_func(self, model, data_loader, device):
        model.eval()
        fitness = 0

        # for idx, (data, target) in enumerate(data_loader):
        data, target = next(iter(data_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)

        # fitness += f1_score(
        #     y_true=target.cpu().numpy(), y_pred=pred.cpu().numpy(), average="macro"
        # )
        fitness += multiclass_f1_score(
            output, target, average="weighted", num_classes=self.num_classes
        ).item()

        return 1 - fitness

    def top1_func(self, model, data_loader, device):
        model.eval()
        fitness = 0

        data, target = next(iter(data_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)

        fitness += top_k_accuracy_score(
            y_true=target.cpu().numpy(),
            y_score=output.cpu().detach().numpy(),
            k=1,
            labels=np.arange(self.num_classes),
        )
        # fitness += topk_multilabel_accuracy(output, target, k=5).item()

        return 1 - fitness

    def test_func(self, X):
        uxi = X.copy()
        if self.block:
            uxi = self.unblocker(uxi)

        self.set_model_state(model=self.model, parameters=uxi)
        self.model.eval()

        fitness = 0
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)

                fitness += multiclass_f1_score(
                    pred, target, average="macro", num_classes=self.num_classes
                ).item()

        return fitness / len(self.test_loader)

    def _calc_pareto_front(self):
        return 0

    def _calc_pareto_set(self):
        return np.full(self.n_var, 0)

    def _evaluate(self, X, out, *args, **kwargs):
        NP = len(X)
        fout = np.zeros(NP)

        for i in range(NP):
            uxi = X[i]
            if self.block and len(uxi) != self.orig_dims:
                uxi = self.unblocker(uxi)

            self.set_model_state(model=self.model, parameters=uxi)

            fitness = self.fitness_func(
                model=self.model, data_loader=self.data_loader, device=self.device
            )
            fout[i] = fitness

        out["F"] = fout

    def scipy_fitness_func(self, X):

        uxi = X
        if self.block and len(uxi) != self.orig_dims:
            uxi = self.unblocker(uxi)

        self.set_model_state(model=self.model, parameters=uxi)

        fitness = self.fitness_func(
            model=self.model, data_loader=self.data_loader, device=self.device
        )

        return fitness

    def multithread_fitness_func(self, X, idx):

        uxi = X
        if self.block and len(uxi) != self.orig_dims:
            uxi = self.unblocker(uxi)

        self.set_model_state(model=self.model, parameters=uxi)

        self.fitness[idx] = self.fitness_func(
            model=self.model, data_loader=self.data_loader, device=self.device
        )

        # return fitness


class SOCallback(Callback):

    def __init__(
        self,
        k_steps=10,
        problem=None,
        csv_path=None,
        plt_path=None,
        start_eval=1,
        start_iter=0,
    ) -> None:
        super().__init__()
        self.k_steps = k_steps
        self.csv_path = csv_path
        self.plt_path = plt_path
        self.problem = problem

        self.data["opt_F"] = []
        self.data["pop_F"] = []
        self.data["n_evals"] = []
        self.start_iter = start_iter
        self.start_eval = start_eval

    def notify(self, algorithm):
        self.data["opt_F"].append(algorithm.opt.get("F")[0][0])
        self.data["pop_F"].append(algorithm.pop.get("F"))
        self.data["n_evals"].append(algorithm.evaluator.n_eval)

        df = pd.read_csv(self.csv_path)
        # if len(df) >= 2:

        if (self.start_iter + algorithm.n_iter) % self.k_steps == 0:
            best_X = algorithm.opt.get("X")[0]

            NP = len(algorithm.pop)
            # algorithm.evaluator.n_eval += NP

            # Define the new row as a dictionary
            new_row = {
                "n_step": self.start_iter + algorithm.n_iter,
                "n_eval": self.start_eval + algorithm.evaluator.n_eval,
                "f_best": algorithm.opt.get("F")[0][0],
                "f_avg": algorithm.pop.get("F").mean(),
                "f_std": algorithm.pop.get("F").std(),
                "test_f1_best": algorithm.problem.test_func(best_X),
            }
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Save the DataFrame back to the CSV file
            df.to_csv(self.csv_path, index=False)

            plt.plot(df["n_step"].to_numpy(), df["f_best"].to_numpy(), label="train")
            if algorithm.problem.criterion != "crossentropy":
                plt.plot(
                    df["n_step"].to_numpy(),
                    1 - df["test_f1_best"].to_numpy(),
                    label="test",
                )
            plt.xlabel("Steps")
            # if self.criterion != "crossentropy":
            plt.ylabel("Error")
            plt.title(f"{algorithm.__class__.__name__}, {algorithm.problem.criterion}")
            plt.legend()
            plt.grid()
            plt.savefig(self.plt_path)
            plt.close()

    def scipy_func(self, intermediate_result):

        if intermediate_result.nit % self.k_steps == 0:
            best_X, best_F = intermediate_result.x, intermediate_result.fun
            df = pd.read_csv(self.csv_path)
            # Define the new row as a dictionary
            new_row = {
                "n_step": intermediate_result.nit,
                "n_eval": intermediate_result.nfev,
                "f_best": best_F,
                "f_avg": intermediate_result.population_energies.mean(),
                "f_std": intermediate_result.population_energies.std(),
                "test_f1_best": self.problem.test_func(best_X),
            }
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Save the DataFrame back to the CSV file
            df.to_csv(self.csv_path, index=False)

            plt.plot(df["n_step"].to_numpy(), df["f_best"].to_numpy(), label="train")
            if self.problem.criterion != "crossentropy":
                plt.plot(
                    df["n_step"].to_numpy(),
                    1 - df["test_f1_best"].to_numpy(),
                    label="test",
                )
            plt.xlabel("Steps")
            # if self.criterion != "crossentropy":
            plt.ylabel("Error")
            plt.title(f"DE, {self.problem.criterion}")
            plt.legend()
            plt.grid()
            plt.savefig(self.plt_path)
            plt.close()

    def general_caller(
        self,
        niter,
        neval,
        opt_X,
        opt_F,
        pop_F,
    ):
        if niter % self.k_steps == 0:
            best_X, best_F = opt_X, opt_F
            df = pd.read_csv(self.csv_path)
            # Define the new row as a dictionary
            new_row = {
                "n_step": self.start_iter + niter,
                "n_eval": self.start_eval + neval,
                "f_best": best_F,
                "f_avg": pop_F.mean(),
                "f_std": pop_F.std(),
                "test_f1_best": self.problem.test_func(best_X),
            }
            # Append the new row to the DataFrame
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            # Save the DataFrame back to the CSV file
            df.to_csv(self.csv_path, index=False)

            plt.plot(df["n_step"].to_numpy(), df["f_best"].to_numpy(), label="train")
            if self.problem.criterion != "crossentropy":
                plt.plot(
                    df["n_step"].to_numpy(),
                    1 - df["test_f1_best"].to_numpy(),
                    label="test",
                )
            plt.xlabel("Steps")
            # if self.criterion != "crossentropy":
            plt.ylabel("Error")
            plt.title(f"DE, {self.problem.criterion}")
            plt.legend()
            plt.grid()
            plt.savefig(self.plt_path)
            plt.close()
