"""P2S (Production Press Sensor Data) time-series classification dataset.

Wraps the gated ``AIML-TUDA/P2S`` dataset on the Hugging Face Hub. P2S contains
force-sensor recordings from a metal stamping / deep-drawing production press;
the task is binary classification — predict whether a press run produced a
*normal* (``0``) or a *defective* (``1``) part from its 4096-step sensor series.

The Hub ships two variants, each with a ``train`` and ``test`` split:

- **Normal** — the honest split (train/test share the same speed distribution).
- **Decoy**  — deliberately correlates production speed with the label to probe
  whether a model latches onto that confounder.

We load **only the Normal variant**. The Hub stores, per row, the sensor series
(``dowel_deep_drawing_ow``, 4096 steps), the label, the run ``speed`` and an
annotation ``mask`` marking speed-affected intervals. We use only the sensor
series and the label; ``speed`` and ``mask`` (the confounder machinery) are
ignored.

Authentication
--------------
This dataset is **gated**: reading it requires accepting the terms on the Hub
and a Hugging Face token in the environment (``HF_TOKEN`` or a cached
``huggingface-cli login``). The dataset does not read tokens itself.

Data contract output
--------------------
X_train : List[np.ndarray (4096, 1)]   one series per training sample
y_train : np.ndarray (N,) int          class labels (0/1)
X_test  : List[np.ndarray (4096, 1)]
y_test  : np.ndarray (M,) int
task    : "classification"
metrics : ["accuracy", "balanced_accuracy", "f1_weighted"]
n_classes : 2
"""

import numpy as np
import pandas as pd
from benchopt import BaseDataset

# Requirement checks
import fsspec  # noqa: F401
from huggingface_hub import hf_hub_download  # noqa: F401

# Columns to read from the parquet: the sensor series and the label. The run
# ``speed`` and annotation ``mask`` columns are the confounder machinery and are
# intentionally skipped (also avoids downloading the large ``mask`` column).
_SENSOR_COL = "dowel_deep_drawing_ow"
_LABEL_COL = "label"

_HF_PARQUET = "hf://datasets/AIML-TUDA/P2S/{variant}/{split}-00000-of-00001.parquet"

# Number of samples kept per split in ``debug`` mode.
_DEBUG_N = 20


class Dataset(BaseDataset):
    """P2S press-sensor binary classification dataset (Normal variant).

    Parameters
    ----------
    variant : str
        Which Hub variant to load. Only ``"Normal"`` is shipped; the path is
        templated so ``"Decoy"`` is a one-line addition if ever needed.
    debug : bool
        If True, keep only the first 20 samples of each split for fast testing.
    """

    name = "P2S"

    requirements = ["pip::huggingface_hub", "fsspec"]

    parameters = {
        "variant": ["Normal"],
        "debug": [False],
    }

    test_parameters = {
        "variant": ["Normal"],
        "debug": [True],
    }

    def _load_split(self, split):
        """Read one split into a list of ``(4096, 1)`` series and int labels."""
        url = _HF_PARQUET.format(variant=self.variant, split=split)
        df = pd.read_parquet(url, columns=[_SENSOR_COL, _LABEL_COL])

        if self.debug:
            df = df.head(_DEBUG_N)

        X = [
            np.asarray(cell, dtype=np.float32).reshape(-1, 1)
            for cell in df[_SENSOR_COL]
        ]
        y = df[_LABEL_COL].to_numpy(dtype=np.int64)
        return X, y

    def get_data(self):
        X_train, y_train = self._load_split("train")
        X_test, y_test = self._load_split("test")

        return dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task="classification",
            metrics=["accuracy", "balanced_accuracy", "f1_weighted"],
            n_classes=2,
        )
