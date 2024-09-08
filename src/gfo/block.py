import numpy as np


def build_rand_blocks(D, bs=100):
    bD = D // block_size + 1
    codebook = {}
    random_blocks = np.random.choice(
        np.arange(bD * block_size), size=(bD, block_size), replace=False
    )
    random_blocks[random_blocks >= D] = -1
    codebook = {i: row[row != -1].tolist() for i, row in enumerate(random_blocks)}

    return codebook


def build_separate_blocks(D, bs1=100, bs2=100, model=None):

    feature_extraction_layers_idx_start, feature_extraction_layers_idx_end = 0, 0
    fc_idx_start, fc_idx_end = 0, 0
    counted_params = 0
    for name, params in model.named_parameters():
        if (
            ("fc" in name or "classifier" in name or "out" in name)
            and counted_params != 0
            and fc_idx_start == 0
        ):
            feature_extraction_layers_idx_end = counted_params - 1
            fc_idx_start = feature_extraction_layers_idx_end + 1
            counted_params = 0
        counted_params += params.size().numel()
    fc_idx_end = counted_params + fc_idx_start

    values = np.arange(
        feature_extraction_layers_idx_start, feature_extraction_layers_idx_end
    )  # Values from 1 to 1223
    np.random.shuffle(values)  # Shuffle values randomly
    rows1, cols1 = (
        feature_extraction_layers_idx_end - feature_extraction_layers_idx_start
    ) // bs1 + 1, bs1

    # Split the array into 12 rows of 100 values and one row with the remaining values
    codebook = {}
    for i in range(rows1):
        start_idx = i * cols1
        end_idx = (i + 1) * cols1
        codebook[i] = (
            values[start_idx:end_idx].tolist()
            if end_idx <= len(values)
            else values[start_idx:].tolist()
        )

    values = np.arange(fc_idx_start, fc_idx_end)  # Values from 1 to 1223
    np.random.shuffle(values)  # Shuffle values randomly
    rows2, cols2 = (fc_idx_end - fc_idx_start) // bs2 + 1, bs2

    # Split the array into 12 rows of 100 values and one row with the remaining values
    for i in range(rows2):
        start_idx = i * cols2
        end_idx = (i + 1) * cols2
        codebook[i + rows1] = (
            values[start_idx:end_idx].tolist()
            if end_idx <= len(values)
            else values[start_idx:].tolist()
        )
    return codebook


def blocker(params, codebook):
    blocked_params = []
    for block_idx, indices in (codebook).items():
        blocked_params.append(params[indices].mean())

    return np.array(blocked_params)
