import numpy as np


def build_rand_blocks(D, block_size=100):
    bD = D // block_size + 1
    codebook = {}
    random_blocks = np.random.choice(
        np.arange(bD * block_size), size=(bD, block_size), replace=False
    )
    random_blocks[random_blocks >= D] = -1
    codebook = {i: row[row != -1].tolist() for i, row in enumerate(random_blocks)}

    return codebook


def blocker(params, codebook):
    blocked_params = []
    for block_idx, indices in (codebook).items():
        blocked_params.append(params[indices].mean())

    return np.array(blocked_params)
