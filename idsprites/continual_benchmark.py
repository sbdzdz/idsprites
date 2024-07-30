"""Class-incremental continual learning dataset."""

from collections import Counter
from itertools import zip_longest

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split

from idsprites import InfiniteDSpritesMap


class ContinualBenchmark:
    def __init__(
        self,
        cfg: DictConfig,
        shapes: list,
        exemplars: list,
        accumulate_test: bool = True,
    ):
        """Initialize the continual learning benchmark.
        Args:
            cfg: The configuration object.
            shapes: The list of shapes.
            exemplars: The list of exemplars.
            accumulate_test: Whether to accumulate the test dataset.
            only_labels: Whether to only return shape labels.
        Returns:
            The continual learning benchmark.
        """
        self.shapes = shapes
        self.shape_ids = range(len(shapes))
        self.exemplars = exemplars
        self.accumulate_test = accumulate_test

        self.factor_resolution = cfg.dataset.factor_resolution
        self.img_size = cfg.dataset.img_size
        self.shapes_per_task = cfg.dataset.shapes_per_task
        self.tasks = cfg.dataset.tasks
        self.train_split = cfg.dataset.train_split
        self.val_split = cfg.dataset.val_split
        self.test_split = cfg.dataset.test_split

        if self.accumulate_test:
            self.test_dataset_size = cfg.dataset.test_dataset_size

    def __iter__(self):
        if self.accumulate_test:
            cumulative_test = BalancedDataset(
                self.test_dataset_size, self.img_size, self.shapes
            )
        for task_shapes, task_shape_ids, task_exemplars in zip(
            self.grouper(self.shapes, self.shapes_per_task),
            self.grouper(self.shape_ids, self.shapes_per_task),
            self.grouper(self.exemplars, self.shapes_per_task),
        ):
            train, val, test = self.build_datasets(task_shapes, task_shape_ids)
            if self.accumulate_test:
                cumulative_test.update(test)
                yield (train, val, cumulative_test.dataset), task_exemplars
            else:
                yield (train, val, test), task_exemplars

    @staticmethod
    def grouper(iterable, n):
        """Iterate in groups of n elements, e.g. grouper(3, 'ABCDEF') --> ABC DEF.
        Args:
            n: The number of elements per group.
            iterable: The iterable to be grouped.
        Returns:
            An iterator over the groups.
        """
        args = [iter(iterable)] * n
        return (list(group) for group in zip_longest(*args))

    def build_datasets(self, shapes: list, shape_ids: list):
        """Build data loaders for a class-incremental continual learning scenario."""
        n = self.factor_resolution
        scale_range = np.linspace(0.5, 1.0, n)
        orientation_range = np.linspace(0, 2 * np.pi * (n / (n + 1)), n)
        position_x_range = np.linspace(0, 1, n)
        position_y_range = np.linspace(0, 1, n)

        dataset = InfiniteDSpritesMap(
            img_size=self.img_size,
            shapes=shapes,
            shape_ids=shape_ids,
            scale_range=scale_range,
            orientation_range=orientation_range,
            position_x_range=position_x_range,
            position_y_range=position_y_range,
        )
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [
                self.train_split,
                self.val_split,
                self.test_split,
            ],
        )
        return train_dataset, val_dataset, test_dataset


class ContinualBenchmarkRehearsal(ContinualBenchmark):
    """Class-incremental continual learning dataset with rehearsal."""

    def __init__(
        self,
        cfg: DictConfig,
        shapes: list,
        exemplars: list,
        only_buffer: bool = False,
    ):
        super().__init__(cfg, shapes, exemplars)
        self.train_dataset_size = cfg.dataset.train_dataset_size
        self.val_dataset_size = cfg.dataset.val_dataset_size
        self.only_buffer = only_buffer

    def __iter__(self):
        train = BalancedDataset(self.train_dataset_size, self.img_size, self.shapes)
        val = BalancedDataset(self.val_dataset_size, self.img_size, self.shapes)
        test = BalancedDataset(self.test_dataset_size, self.img_size, self.shapes)
        for task_shapes, task_shape_ids, task_exemplars in zip(
            self.grouper(self.shapes, self.shapes_per_task),
            self.grouper(self.shape_ids, self.shapes_per_task),
            self.grouper(self.exemplars, self.shapes_per_task),
        ):
            train_task, val_task, test_task = self.build_datasets(
                task_shapes, task_shape_ids
            )
            train.update(train_task)
            val.update(val_task)
            test.update(test_task)

            if self.only_buffer:
                train_dataset = train.dataset
            else:
                train_dataset = self.mix_in_current_data(train, train_task)

            yield (train_dataset, val.dataset, test.dataset), task_exemplars

    def mix_in_current_data(self, train, train_task):
        """Mix in current data with the buffer.
        Randomly sample half of the data from the buffer and half from the current task.
        Args:
            train: The current buffer.
            train_task: The current task.
        Returns:
            The mixed-in dataset.
        """
        num_buffer_samples = len(train_task.indices) // 2
        buffer_indices = np.random.choice(
            len(train.dataset), num_buffer_samples, replace=True
        )
        buffer_samples = [train.dataset.data[idx] for idx in buffer_indices]

        num_task_samples = len(train_task.dataset) - num_buffer_samples
        task_indices = np.random.choice(
            train_task.indices, num_task_samples, replace=False
        )
        task_samples = [train_task.dataset.data[idx] for idx in task_indices]

        train_dataset = InfiniteDSpritesMap(
            img_size=self.img_size,
            dataset_size=1,
            shapes=self.shapes,
        )
        train_dataset.data = buffer_samples + task_samples
        return train_dataset


class BalancedDataset:
    """Class-balanced reservoir sampling."""

    def __init__(self, max_size: int, img_size: int, shapes: list) -> None:
        """Initialize the class-balanced reservoir sampling.
        Args:
            max_size: The maximum size of the dataset.
        """
        self.max_size = max_size
        self.img_size = img_size
        self.shapes = shapes

        self.dataset = InfiniteDSpritesMap(
            img_size=self.img_size,
            dataset_size=1,
            shapes=self.shapes,
        )
        self.dataset.data = []  # dummy dataset
        self.stored_class_counts = Counter()
        self.seen_class_counts = Counter()
        self.full_classes = set()

    def update(self, task_dataset: Dataset) -> None:
        task_data = [task_dataset.dataset.data[idx] for idx in task_dataset.indices]

        for factors in task_data:
            shape_id = factors.shape_id
            if len(self.dataset) < self.max_size:
                self.dataset.data.append(factors)
                self.stored_class_counts[shape_id] += 1
            else:  # buffer is full
                self.full_classes.update(self.get_largest_classes())
                if shape_id in self.full_classes:
                    u = np.random.uniform()
                    stored = self.stored_class_counts[shape_id]
                    seen = self.seen_class_counts[shape_id]
                    if u <= stored / seen:
                        idx = self.get_random_instance(shape_id)
                        self.dataset.data[idx] = factors
                else:  # replace a random instance of the largest class
                    largest_class = np.random.choice(self.get_largest_classes())
                    idx = self.get_random_instance(largest_class)
                    self.dataset.data[idx] = factors
                    self.stored_class_counts[shape_id] += 1
                    self.stored_class_counts[largest_class] -= 1
            self.seen_class_counts[shape_id] += 1

    def get_largest_classes(self):
        """Get the most represented classes."""
        max_count = max(self.stored_class_counts.values())
        return [
            class_id
            for class_id, count in self.stored_class_counts.items()
            if count == max_count
        ]

    def get_random_instance(self, class_id: int):
        """Return the index of a random instance of a given class."""
        data = self.dataset.data
        indices = [i for i, factors in enumerate(data) if factors.shape_id == class_id]
        return np.random.choice(indices)
