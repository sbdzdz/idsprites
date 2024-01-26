import argparse
from itertools import zip_longest
from pathlib import Path

from PIL import Image
from torch.utils.data import random_split

import idsprites as ids
import numpy as np


def generate_dataset(args):
    dataset = ids.InfiniteDSprites(img_size=args.img_size)
    num_shapes = args.num_tasks * args.shapes_per_task
    shapes = [dataset.generate_shape() for _ in range(num_shapes)]
    exemplars = generate_exemplars(shapes, args.img_size)
    shape_ids = list(range(num_shapes))

    for task, task_shapes, task_shape_ids, task_exemplars in zip(
        range(args.num_tasks),
        grouper(shapes, args.shapes_per_task),
        grouper(shape_ids, args.shapes_per_task),
        grouper(exemplars, args.shapes_per_task),
    ):
        task_dir = args.out_dir / f"task_{task + 1}"
        exemplars_dir = task_dir / "exemplars"
        train_dir = task_dir / "train"
        val_dir = task_dir / "val"
        test_dir = task_dir / "test"

        for subdir in [exemplars_dir, train_dir, val_dir, test_dir]:
            subdir.mkdir(parents=True, exist_ok=True)

        dataset = ids.ContinualDSpritesMap(
            img_size=args.img_size, shapes=task_shapes, shape_ids=task_shape_ids
        )
        for i, exemplar in enumerate(task_exemplars):
            exemplar = to_image(exemplar)
            exemplar.save(exemplars_dir / f"exemplar_{i}.png")

        train, val, test = random_split(
            dataset, [args.train_split, args.val_split, args.test_split]
        )
        for split, subdir in zip([train, val, test], [train_dir, val_dir, test_dir]):
            for i, (image, factors) in enumerate(split):
                image = to_image(image)
                image.save(subdir / f"sample_{i}_shape_{factors.shape_id}.png")


def to_image(array: np.ndarray):
    """Convert a tensor to an image."""
    array = array.transpose(1, 2, 0)
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


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


def generate_exemplars(shapes, img_size: int):
    """Generate a batch of exemplars for training and visualization."""
    dataset = ids.InfiniteDSprites(
        img_size=img_size,
    )
    return [
        dataset.draw(
            ids.Factors(
                color=(1.0, 1.0, 1.0),
                shape=shape,
                shape_id=None,
                scale=1.0,
                orientation=0.0,
                position_x=0.5,
                position_y=0.5,
            ),
        )
        for shape in shapes
    ]


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=root / "data")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--shapes_per_task", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--train_split", type=int, default=0.8)
    parser.add_argument("--val_split", type=int, default=0.1)
    parser.add_argument("--test_split", type=int, default=0.1)
    args = parser.parse_args()

    generate_dataset(args)
