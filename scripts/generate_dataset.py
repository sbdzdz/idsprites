import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import random_split
from tqdm import tqdm

import idsprites as ids


def process_task(task, task_shapes, task_shape_ids, task_exemplars, args):
    if not args.overwrite and (args.out_dir / f"task_{task + 1}").exists():
        return
    task_dir = args.out_dir / f"task_{task + 1}"
    exemplars_dir = task_dir / "exemplars"
    train_dir = task_dir / "train"
    val_dir = task_dir / "val"
    test_dir = task_dir / "test"

    for subdir in [exemplars_dir, train_dir, val_dir, test_dir]:
        subdir.mkdir(parents=True, exist_ok=True)

    np.save(task_dir / "shapes.npy", task_shapes)
    for i, exemplar in enumerate(task_exemplars):
        exemplar = to_image(exemplar)
        exemplar.save(exemplars_dir / f"exemplar_{i}.png")

    dataset = create_dataset(args, task_shapes, task_shape_ids)

    train, val, test = random_split(
        dataset, [args.train_split, args.val_split, args.test_split]
    )
    for split, subdir in zip([train, val, test], [train_dir, val_dir, test_dir]):
        split_factors = []
        labels = []
        for i, (image, factors) in enumerate(split):
            split_factors.append(factors.replace(shape=None))
            image = to_image(image)
            path = subdir / f"sample_{i}.png"
            image.save(path)
            labels.append(f"{path.name} {factors.shape_id}")
        split_factors = ids.Factors(*zip(*split_factors))
        np.savez(subdir / "factors.npz", **split_factors._asdict())
        with open(subdir / "labels.txt", "w") as f:
            f.write("\n".join(labels))


def generate_dataset(args):
    dataset = ids.InfiniteDSprites(img_size=args.img_size)
    num_shapes = args.num_tasks * args.shapes_per_task
    shapes = [dataset.generate_shape() for _ in range(num_shapes)]
    exemplars = generate_exemplars(shapes, args.img_size)
    shape_ids = list(range(num_shapes))

    tasks = list(range(args.num_tasks))
    task_shapes = list(grouper(shapes, args.shapes_per_task))
    task_shape_ids = list(grouper(shape_ids, args.shapes_per_task))
    task_exemplars = list(grouper(exemplars, args.shapes_per_task))

    # parallelize over tasks
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    process_task,
                    tasks,
                    task_shapes,
                    task_shape_ids,
                    task_exemplars,
                    [args] * args.num_tasks,
                ),
                total=args.num_tasks,
            )
        )


def create_dataset(args, task_shapes, task_shape_ids):
    """Create a dataset for a single task."""
    n = args.factor_resolution
    scale_range = np.linspace(0.5, 1.0, n)
    orientation_range = np.linspace(0, 2 * np.pi * (n) / (n + 1), n)
    position_x_range = np.linspace(0, 1, n)
    position_y_range = np.linspace(0, 1, n)
    dataset = ids.InfiniteDSpritesMap(
        img_size=args.img_size,
        scale_range=scale_range,
        orientation_range=orientation_range,
        position_x_range=position_x_range,
        position_y_range=position_y_range,
        shapes=task_shapes,
        shape_ids=task_shape_ids,
    )
    return dataset


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
                shape=shape,
                color=(1.0, 1.0, 1.0),
                shape_id=None,
                scale=1.0,
                orientation=0.0,
                position_x=0.5,
                position_y=0.5,
            ),
        )
        for shape in shapes
    ]


def int_or_float(value):
    """Convert a string to an int or float."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError as err:
            raise argparse.ArgumentTypeError(
                f"{value} is not a valid int or float"
            ) from err


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=root / "data")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--shapes_per_task", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--train_split", type=int_or_float, default=0.98)
    parser.add_argument("--val_split", type=int_or_float, default=0.01)
    parser.add_argument("--test_split", type=int_or_float, default=0.01)
    parser.add_argument("--factor_resolution", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate_dataset(args)
