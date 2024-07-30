"""Visualization utilities for the dSprites and InfiniteDSprites datasets.
Example:
    python -c "from codis.data.visualization import draw_shapes; draw_shapes()"
"""

import io
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import PIL
from matplotlib import pyplot as plt
from tqdm import tqdm

from idsprites import (
    InfiniteDSprites,
    Factors,
)

repo_root = Path(__file__).parent.parent

COLORS = [
    "white",
    "whitesmoke",
    "purple",
    "maroon",
    "darkblue",
    "teal",
    "peachpuff",
    "darkgreen",
]


def draw_batch(
    images,
    path: Path = repo_root / "img/batch_grid.png",
    fig_height: float = 10,
    num_images: int = 16,
    save: bool = True,
    show: bool = False,
):
    """Show a batch of images on a grid.
    Only the first n_max images are shown.
    Args:
        images: A numpy array of shape (N, C, H, W) or (N, H, W)
        path: The path to save the image to
        fig_height: The height of the figure in inches
        num_images: The maximum number of images to show
        show: Whether to show the image
    Returns:
        None
    """
    num_images = min(images.shape[0], num_images)
    if images.ndim == 4:
        images = np.transpose(images, (0, 2, 3, 1))
    nrows = int(np.ceil(np.sqrt(num_images)))
    ncols = int(np.ceil(num_images / nrows))
    _, axes = plt.subplots(
        nrows, ncols, figsize=(ncols / nrows * fig_height, fig_height)
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax in axes.flat:
        ax.axis("off")

    for ax, img in zip(axes.flat, images[:num_images]):
        ax.imshow(img, cmap="Greys_r", aspect="equal")

    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    buffer = io.BytesIO()
    plt.savefig(buffer, bbox_inches="tight")
    plt.close()

    return PIL.Image.open(buffer)


def draw_batch_and_reconstructions(
    *image_arrays,
    fig_height: float = 10,
    num_images: int = 25,
    path: Path = repo_root / "img/reconstructions.png",
    save=True,
    show=False,
):
    """Show a batch of images and their reconstructions on a grid.
    Only the first n_max images are shown.
    Args:
        image_arrays: Numpy arrays of shape (N, C, H, W) or (N, H, W)
        fig_height: The height of the figure in inches
        num_images: The maximum number of images to show
        path: The path to save the image to
        show: Whether to show the image
    Returns:
        None
    """
    img = image_arrays[0]
    num_images = min(img.shape[0], num_images)
    if img.ndim == 4:
        image_arrays = [np.transpose(img, (0, 2, 3, 1)) for img in image_arrays]
    nrows = int(np.ceil(np.sqrt(num_images)))
    ncols = int(np.ceil(num_images / nrows))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(len(image_arrays) * ncols / nrows * fig_height, fig_height),
    )
    fig.tight_layout()
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax in axes.flat:
        ax.axis("off")

    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            break
        images = [img[i] for img in image_arrays]
        concatenated = np.concatenate(images, axis=1)
        border_width = concatenated.shape[1] // 128 or 1

        for j in range(1, len(image_arrays)):
            mid = j * concatenated.shape[1] // len(image_arrays)
            concatenated[:, mid - border_width : mid + border_width] = 1.0
        ax.imshow(concatenated, cmap="Greys_r", aspect="equal")
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    buffer = io.BytesIO()
    plt.savefig(buffer, bbox_inches="tight")
    plt.close()

    return PIL.Image.open(buffer)


def draw_shapes(
    path: Path = repo_root / "img/shapes.png",
    nrows: int = 5,
    ncols: int = 12,
    fig_height: float = 10,
    img_size: int = 128,
    frame_color: str = "darkgray",
    background_color: str = "lightgray",
    orientation_marker_color: str = "black",
    seed: int = 0,
    fill_shape: bool = True,
    debug: bool = False,
    canonical: bool = True,
):
    """Plot an n x n grid of random shapes.
    Args:
        path: The path to save the image to
        nrows: The number of rows in the grid
        ncols: The number of columns in the grid
        fig_height: The height of the figure in inches
        img_size: The size of the image in pixels
        frame_color: The color of the frame
        fg_color: The color of the shape
        bg_color: The color of the background plot area
        seed: The random seed to use
        fill_shape: Whether to fill the shape or just draw the outline
        debug: Whether to draw additional debug info
        canonical: Whether to draw the shape in canonical form.
    Returns:
        None
    """
    path = Path(path)
    np.random.seed(seed)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols / nrows * fig_height, fig_height),
        layout="tight",
        subplot_kw={"aspect": 1.0},
        facecolor=frame_color,
    )
    fig.tight_layout()
    dataset = InfiniteDSprites(
        img_size=img_size,
        color_range=COLORS,
        background_color=background_color,
        orientation_marker_color=orientation_marker_color,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax in axes.flat:
        shape = dataset.generate_shape()
        ax.axis("off")
        if fill_shape:
            if canonical:
                factors = Factors(
                    shape=shape,
                    color=dataset.sample_factors().color,
                    shape_id=None,
                    scale=1.0,
                    orientation=0.0,
                    position_x=0.5,
                    position_y=0.5,
                )
            else:
                factors = dataset.sample_factors().replace(shape=shape)
            img = dataset.draw(factors, channels_first=False, debug=debug)
            ax.imshow(img, cmap="Greys_r", aspect="equal")
        else:
            ax.plot(shape[0], shape[1], color=dataset.sample_factors().color)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)


def draw_shapes_animated(
    path: Path = repo_root / "img/shapes.gif",
    nrows: int = 5,
    ncols: int = 12,
    fig_height: float = 10,
    img_size: int = 256,
    frame_color: str = "darkgray",
    background_color: str = "lightgray",
    orientation_marker_color: str = "black",
    duration: int = 8,
    fps: int = 60,
    factor: str = None,
    seed: int = 0,
    debug: bool = False,
):
    """Create an animated GIF showing a grid of shapes undergoing transformations.
    Args:
        path: The path to save the image to.
        nrows: The number of rows in the grid.
        ncols: The number of columns in the grid.
        fig_height: The height of the figure in inches.
        img_size: The size of the image in pixels.
        frame_color: The color of the background plot area.
        background_color: The color of the canvas area.
        duration: The duration of the animation in seconds.
        fps: The number of frames per second.
        factor: The factor to vary. If None, all factors are varied.
        seed: The random seed.
        debug: Whether to draw additional debug info.
    Returns:
        None
    """
    path = Path(path)
    np.random.seed(seed)
    num_frames = fps * duration
    dataset = InfiniteDSprites(
        img_size=img_size,
        color_range=COLORS,
        scale_range=np.linspace(0.1, 0.8, num_frames // 4),
        orientation_range=np.linspace(0.0, 2 * np.pi, num_frames // 4),
        position_x_range=np.linspace(0.0, 1.0, num_frames // 4),
        position_y_range=np.linspace(0.0, 1.0, num_frames // 4),
        background_color=background_color,
        orientation_marker_color=orientation_marker_color,
    )
    shapes = [dataset.generate_shape() for _ in range(nrows * ncols)]
    colors = [dataset.sample_factors().color for _ in range(nrows * ncols)]
    if factor is None:
        factors = generate_multi_factor_sequence(dataset)
    else:
        path = path.with_stem(f"{path.stem}_{factor}")
        factors = generate_single_factor_sequence(dataset, factor)

    frames = [
        [
            dataset.draw(
                Factors(
                    shape=shape,
                    color=color,
                    shape_id=None,
                    scale=scale,
                    orientation=orientation,
                    position_x=position_x,
                    position_y=position_y,
                ),
                channels_first=False,
                debug=debug,
            )
            for shape, color in zip(shapes, colors)
        ]
        for scale, orientation, position_x, position_y in zip(*factors)
    ]
    save_animation(path, frames, nrows, ncols, fig_height, frame_color, fps)


def generate_multi_factor_sequence(dataset):
    """Generate a sequence of factors that can be used to animate a shape.
    Args:
        dataset: The dataset to generate the factors for.
    Returns:
        A tuple of factor value sequences representing a smooth animation.
    """
    scale_range, orientation_range, position_x_range, position_y_range = (
        dataset.ranges["scale"],
        dataset.ranges["orientation"],
        dataset.ranges["position_x"],
        dataset.ranges["position_y"],
    )
    length = (
        len(scale_range)
        + len(orientation_range)
        + len(position_x_range)
        + len(position_y_range)
    )
    scales, orientations, positions_x, positions_y = (
        np.zeros(length),
        np.zeros(length),
        np.zeros(length),
        np.zeros(length),
    )

    start = 0
    scales[start : len(scale_range)] = scale_range
    scales[len(scale_range) :] = scale_range[-1]

    start = len(scale_range)
    orientations[start : start + len(orientation_range)] = orientation_range
    orientations[start + len(orientation_range) :] = orientation_range[-1]

    start = len(scale_range) + len(orientation_range)
    positions_x[start : start + len(position_x_range)] = position_x_range
    positions_x[start + len(position_x_range) :] = position_x_range[-1]

    start = len(scale_range) + len(orientation_range) + len(position_x_range)
    positions_y[start : start + len(position_y_range)] = position_y_range
    positions_y[start + len(position_y_range) :] = position_y_range[-1]
    return scales, orientations, positions_x, positions_y


def generate_single_factor_sequence(dataset, factor):
    """Generate a smooth progression of a single factor."""
    length = 2 * len(dataset.ranges[factor])
    factors = {
        "scale": np.ones(length) * 0.8,
        "orientation": np.ones(length) * 0.0,
        "position_x": np.ones(length) * 0.5,
        "position_y": np.ones(length) * 0.5,
    }
    factors[factor][: length // 2] = dataset.ranges[factor]
    factors[factor][length // 2 :] = dataset.ranges[factor][::-1]
    return (
        factors["scale"],
        factors["orientation"],
        factors["position_x"],
        factors["position_y"],
    )


def draw_shape_interpolation(
    path: Path = repo_root / "img/smooth_shapes.gif",
    nrows: int = 5,
    ncols: int = 12,
    fig_height: float = 10,
    img_size: int = 256,
    frame_color: str = "darkgray",
    background_color="lightgray",
    orientation_marker_color: str = "black",
    num_shapes: int = 10,
    duration_per_shape: int = 2,
    fps: int = 60,
    seed: int = 0,
):
    """Smoothly interpolate between shapes and colors.
    Args:
        path: The path to save the animation to
        nrows: The number of rows in the grid
        ncols: The number of columns in the grid
        fig_height: The height of the figure in inches
        img_size: The size of the image in pixels
        bg_color: The color of the background plot area
        num_shapes: The number of shapes to interpolate between
        duration_per_shape: The number of seconds per shape transition
        fps: The number of frames per second
        seed: The random seed
    """
    path = Path(path)
    np.random.seed(seed)
    dataset = InfiniteDSprites(
        img_size=img_size,
        color_range=COLORS,
        background_color=background_color,
        orientation_marker_color=orientation_marker_color,
    )
    colors = [
        [dataset.sample_factors().color for _ in range(num_shapes)]
        for _ in range(nrows * ncols)
    ]
    shapes = [
        [dataset.generate_shape() for _ in range(num_shapes)]
        for _ in range(nrows * ncols)
    ]
    shape_sequences = interpolate(shapes, num_frames=fps * duration_per_shape)
    color_sequences = interpolate(colors, num_frames=fps * duration_per_shape)

    frames = [
        [
            dataset.draw(
                Factors(
                    shape=shape,
                    color=color,
                    shape_id=None,
                    scale=1.0,
                    orientation=0.0,
                    position_x=0.5,
                    position_y=0.5,
                ),
                channels_first=False,
            )
            for shape, color in zip(shape_sequence, color_sequence)
        ]
        for shape_sequence, color_sequence in zip(shape_sequences, color_sequences)
    ]
    save_animation(path, frames, nrows, ncols, fig_height, frame_color, fps)


def interpolate(values, num_frames):
    """Interpolate between a sequence of values."""
    sequences = []
    for value in values:
        value.append(value[0])
        sequence = []
        for start, end in zip(value[:-1], value[1:]):
            sequence.extend(np.linspace(start, end, num_frames))
        sequences.append(sequence)

    return zip(*sequences)


def draw_orientation_normalization(
    path: Path = repo_root / "img/orientation_normalization.gif",
    nrows: int = 5,
    ncols: int = 12,
    fig_height: float = 10,
    img_size: int = 256,
    bg_color: str = "white",
    duration: int = 6,
    fps: int = 60,
    seed: int = 0,
):
    """Create an animated GIF showing a grid of shapes undergoing orientation normalization.
    Args:
        path: The path to save the image to.
        nrows: The number of rows in the grid.
        ncols: The number of columns in the grid.
        fig_height: The height of the figure in inches.
        img_size: The size of the image in pixels.
        bg_color: The color of the background plot area.
        duration: The duration of the animation in seconds.
        fps: The number of frames per second.
        seed: The random seed.
    Returns:
        None
    """
    path = Path(path)
    np.random.seed(seed)
    num_frames = fps * duration
    num_shapes = nrows * ncols

    dataset = InfiniteDSprites(
        img_size=img_size,
        color_range=["snow"],
        orientation_range=np.linspace(0.2 * np.pi, 1.8 * np.pi, 32),
    )

    start_factors = [dataset.sample_factors() for _ in range(num_shapes)]
    sequence = []
    for factor in start_factors:
        factors = [
            Factors(
                shape=factor.shape,
                color=factor.color,
                shape_id=None,
                scale=scale,
                orientation=orientation,
                position_x=position_x,
                position_y=position_y,
            )
            for scale, orientation, position_x, position_y in generate_normalization_sequence(
                factor, num_frames
            )
        ]
        sequence.append(factors)
    sequence = list(zip(*sequence))  # transpose the nested list

    frames = [
        [dataset.draw(factor, channels_first=False) for factor in factors]
        for factors in sequence
    ]
    save_animation(path, frames, nrows, ncols, fig_height, bg_color, fps)


def generate_normalization_sequence(factor, num_frames):
    """Generate a sequence of factors"""
    scales, orientations, positions_x, positions_y = (
        2.0 * np.ones(num_frames),
        factor.orientation * np.ones(num_frames),
        0.5 * np.ones(num_frames),
        0.5 * np.ones(num_frames),
    )
    chunk = num_frames // 4

    if factor.orientation <= np.pi:
        orientations[:chunk] = np.linspace(factor.orientation, 0.0, chunk)
        orientations[2 * chunk : 3 * chunk] = np.linspace(
            0.0, factor.orientation, chunk
        )
    else:
        orientations[:chunk] = np.linspace(factor.orientation, 2 * np.pi, chunk)
        orientations[2 * chunk : 3 * chunk] = np.linspace(
            2 * np.pi, factor.orientation, chunk
        )
    orientations[chunk : 2 * chunk] = 0.0
    orientations[3 * chunk :] = factor.orientation

    return zip(scales, orientations, positions_x, positions_y)


def save_animation(path, frames, nrows, ncols, fig_height, bg_color, fps):
    """Save an animation to a GIF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, mode="I", fps=fps) as writer:
        for frame in tqdm(frames):
            _, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols / nrows * fig_height, fig_height),
                layout="tight",
                subplot_kw={"aspect": 1.0},
                facecolor=bg_color,
            )
            buffer = io.BytesIO()
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])

            for ax, image in zip(axes.flat, frame):
                ax.axis("off")
                ax.imshow(image, cmap="Greys_r")
            plt.savefig(buffer, format="png")
            plt.close()
            writer.append_data(imageio.imread(buffer))  # type: ignore
