"""RLDS dataset of Isaac Sim observations paired with LIBERO-Spatial human-demo actions (v4b).

v4b (2026-09-30) differs from the September build only in its source folder and data:
  - 429 human-demo replays (task 5 repeated 3×, 457 files), converted by build_finetune_dataset_v4b.py
  - image k is the replay frame k+1: in replay mode the robot moves to demo pose k during step k and the
    frame is captured at the start of the step (verified with verify_dataset_v4c.py)
  - image preprocessing: square → centre area 0.389 → 224 BILINEAR (with the 0.9 training crop augmentation,
    the average view equals the evaluation server's crop 0.35)

The schema and the dataset name are identical to the September `isaac_libero_spatial`, which the OpenVLA repo
already registers with `libero_dataset_transform` (gripper −1/+1 → clip to [0, 1] → invert). Build it into a
separate data dir so the September build is not touched:

    cd <this directory>
    ISAAC_LIBERO_NPZ_ROOT=$HOME/datasets/finetune_dataset_v4b tfds build --data_dir $HOME/tensorflow_datasets_v4b
"""

import glob
import os
from typing import Any, Iterator

import numpy as np
import tensorflow_datasets as tfds

DATASET_ROOT = os.path.expanduser(os.environ.get("ISAAC_LIBERO_NPZ_ROOT", "~/datasets/finetune_dataset_v4b"))
IMAGE_SIZE = 224
ACTION_DIM = 7
STATE_DIM = 8


class IsaacLiberoSpatial(tfds.core.GeneratorBasedBuilder):
    """Isaac Sim renders of LIBERO-Spatial human-demo replays (v4b)."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "v4b: 429 human-demo replays (457 files, task 5 x3), image offset +1, crop 0.389."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8, encoding_format="png",
                                                     doc="Isaac D405 agentview: square -> centre area 0.389 -> 224 BILINEAR."),
                        "state": tfds.features.Tensor(shape=(STATE_DIM,), dtype=np.float32,
                                                      doc="Isaac eef pos, axis-angle, finger joints. Unused (load_proprio=False)."),
                    }),
                    "action": tfds.features.Tensor(shape=(ACTION_DIM,), dtype=np.float32,
                                                   doc="LIBERO delta EEF action; gripper -1 open / +1 close."),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "task_id": tfds.features.Scalar(dtype=np.int32),
                    "source": tfds.features.Text(),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        paths = sorted(glob.glob(f"{DATASET_ROOT}/task_*/*.npz"))
        if len(paths) != 457:
            raise RuntimeError(f"expected 457 npz under {DATASET_ROOT}, found {len(paths)}")
        return {"train": self._generate_examples(paths)}

    def _generate_examples(self, paths: list) -> Iterator[tuple]:
        for path in paths:
            with np.load(path, allow_pickle=True) as data:
                images = np.asarray(data["image"], dtype=np.uint8)
                actions = np.asarray(data["action"], dtype=np.float32)
                state = np.asarray(data["state"], dtype=np.float32)
                instruction = str(data["instruction"])
                task_id = int(np.asarray(data["task_id"]).reshape(-1)[0])
                source = str(data["source"]) if "source" in data.files else ""
            n = len(actions)
            assert images.shape == (n, IMAGE_SIZE, IMAGE_SIZE, 3), (path, images.shape)
            assert state.shape == (n, STATE_DIM), (path, state.shape)
            assert np.all(np.isin(np.round(actions[:, 6], 3), [-1.0, 1.0])), f"gripper not +-1: {path}"
            steps = [{
                "observation": {"image": images[i], "state": state[i]},
                "action": actions[i],
                "discount": 1.0,
                "reward": float(i == n - 1),
                "is_first": i == 0,
                "is_last": i == n - 1,
                "is_terminal": i == n - 1,
                "language_instruction": instruction,
            } for i in range(n)]
            key = os.path.relpath(path, DATASET_ROOT)
            yield key, {"steps": steps, "episode_metadata": {"file_path": key, "task_id": task_id, "source": source}}
