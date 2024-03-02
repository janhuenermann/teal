from functools import cached_property
from typing import Dict

import numpy as np
from numba import njit


_EPISODE_IDX = 0
_FRAME_IDX = 1
_LAST_IDX = 2
_NEXT_IDX = 3


class Buffer:
    def __init__(self, capacity: int, seed=None):
        self.contents: Dict[str, np.ndarray] = {}

        # _idx stores [episode index, frame index, last frame index] tuple for each batch element
        self._idx: np.ndarray = np.zeros((0, 3), dtype=np.int64)

        self.capacity = capacity
        self.size = 0
        self.pointer = 0
        self.episode_counter = 0
        self.frame_counter = 0

        seeder = np.random.SeedSequence(seed)
        self.rng = np.random.default_rng(seeder)

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return self.contents[key][: self.size]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        if value.shape[0] != self.capacity:
            raise ValueError(
                f"Expected array of shape {(self.capacity, *value.shape[1:])}, "
                f"got {value.shape}"
            )
        if key in self.contents and self.contents[key].shape != value.shape:
            raise ValueError(
                f"Expected array of shape {self.contents[key].shape}, got {value.shape}"
            )
        self.contents[key] = value

    def insert(self, data: Dict[str, np.ndarray]):
        """
        Adds frames to the buffer.

        Args:
            data: A dictionary of numpy arrays. Each array should have the same batch size.
                Must contain either `is_first` or `is_done` as a key.
        """
        assert len(data) > 0
        if "is_done" not in data and "is_first" not in data:
            raise ValueError(
                f"Cannot determine episode boundaries. Please include `is_first` or `is_done` in data. "
                f"Got keys: {data.keys()}"
            )
        if len(self.contents) == 0:
            self._lazy_init(data)
        if "is_first" in data:
            self._end_sequences(np.argwhere(data["is_first"]))

        batch_size = next(iter(data.values())).shape[0]
        indices = np.arange(self.pointer, self.pointer + batch_size) % self.capacity
        for key, value in data.items():
            self.contents[key][indices] = value

        # update last of next index
        next_indices = self.contents["_index"][indices, _NEXT_IDX]
        next_mask = next_indices >= 0
        self.contents["_index"][next_indices[next_mask], _LAST_IDX] = -1

        # update next of last index
        last_indices = self._idx[:, _LAST_IDX]
        last_mask = last_indices >= 0
        self.contents["_index"][last_indices[last_mask], 3] = indices[last_mask]
        self.contents["_index"][indices, :3] = self._idx
        self.contents["_index"][indices, _NEXT_IDX] = -1

        # update data pointers
        self.pointer = (self.pointer + batch_size) % self.capacity
        self.frame_counter += batch_size
        self.size = min(self.size + batch_size, self.capacity)

        # update _idx
        self._idx[:, _FRAME_IDX] += 1  # increment sequence number
        self._idx[:, _LAST_IDX] = indices  # update last index
        if "is_done" in data:
            self._end_sequences(np.argwhere(data["is_done"]))

    def sample_frames(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return self.take_indices(indices)

    def sample_sequence_indices(self, batch_size: int, sequence_size: int):
        start_indices = self.rng.choice(self.size, size=batch_size, replace=False)
        next_indices = self.contents["_index"][:, _NEXT_IDX]
        sequence_indices = walk(start_indices, next_indices, sequence_size)
        valid = sequence_indices[:, :-1] != sequence_indices[:, 1:]
        valid = np.concatenate([np.ones((batch_size, 1), dtype=bool), valid], axis=1)
        return sequence_indices, valid

    def sample_sequences(self, batch_size: int, sequence_size: int):
        sequence_indices, valid = self.sample_sequence_indices(
            batch_size, sequence_size
        )
        batch = self.take_indices(sequence_indices)
        batch["valid"] = valid
        return batch

    def take_indices(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Takes a subset of frames from the buffer. Returns a dictionary of numpy arrays
        with the first dimension being the batch dimension.
        """
        return {key: value[indices] for key, value in self.contents.items()}

    def _lazy_init(self, data: Dict[str, np.ndarray]):
        assert len(data) > 0

        batch_size = next(iter(data.values())).shape[0]
        self._idx = np.zeros((batch_size, 3), dtype=np.int64)
        self._idx[:, _EPISODE_IDX] = np.arange(batch_size)
        self._idx[:, _LAST_IDX] = -1
        self.episode_counter = batch_size - 1

        for key, value in data.items():
            assert (
                value.shape[0] == batch_size
            ), f"Expected batch size {batch_size} for key {key}, "
            self.contents[key] = np.zeros(
                shape=(self.capacity, *value.shape[1:]), dtype=value.dtype
            )

        self.contents["_index"] = np.zeros(shape=(self.capacity, 4), dtype=np.int64)

    def _end_sequences(self, done_indices: np.ndarray) -> None:
        self._idx[done_indices, _FRAME_IDX] = 0  # reset sequence number
        self._idx[done_indices, _LAST_IDX] = -1
        for done_ind in done_indices:
            self._idx[done_ind, _EPISODE_IDX] = self.episode_counter = (
                self.episode_counter + 1
            )

    def linearize(self) -> "PermutedBuffer":
        """
        Brings the buffer into episode major order, so that all frames
        from one episode are contiguous.
        """
        episode_index, frame_index = self.contents["_index"][
            : self.size, (_EPISODE_IDX, _FRAME_IDX)
        ].T
        permutation = np.lexsort((frame_index, episode_index))
        return PermutedBuffer(buffer=self, permutation=permutation)


class PermutedBuffer:
    def __init__(self, buffer: Buffer, permutation: np.ndarray):
        self.buffer = buffer
        self.permutation = permutation
        self._cache: Dict[str, np.ndarray] = {}
        self._state = buffer.frame_counter

    def __getitem__(self, key):
        if self.buffer.frame_counter != self._state:
            raise RuntimeError(
                "Buffer has been modified since the creation of this view. "
                "Please create a new view."
            )
        if key not in self._cache:
            arr = self.buffer.contents[key][self.permutation]
            arr.flags.writeable = False
            self._cache[key] = arr
        return self._cache[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        assert len(self.permutation) == len(
            self.buffer
        ), f"Partial buffer updates from a view are not supported. "
        assert value.shape[0] == len(self.buffer)
        self.buffer[key] = value[self._reverse_permutation]

    @cached_property
    def _reverse_permutation(self):
        return np.argsort(self.permutation)


def compute_buffer_stats(buffer: Buffer):
    reward_avg = buffer["reward"].mean()
    reward_std = buffer["reward"].std()
    buffer_length = len(buffer)
    return dict(
        reward_avg=reward_avg, reward_std=reward_std, buffer_length=buffer_length
    )


@njit
def walk(start_indices, next_indices, length: int):
    """
    Retrieves sequences of length `length` from the buffer.
    """

    batch_size = len(start_indices)
    result = np.full((batch_size, length), -1, dtype=np.int64)
    result[:, 0] = start_indices
    for i in range(batch_size):
        idx = start_indices[i]
        for j in range(1, length):
            next_idx = next_indices[idx]
            if next_idx < 0:
                result[i, j:] = idx
                break
            result[i, j] = next_idx
            idx = next_idx
    return result
