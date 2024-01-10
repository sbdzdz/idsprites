"""Test the infinite dSprites dataset."""
import pytest
import numpy as np

from idsprites import (
    InfiniteDSprites,
    InfiniteDSpritesAnalogies,
    InfiniteDSpritesTriplets,
)


@pytest.mark.parametrize(
    "dataset_class",
    [InfiniteDSprites, InfiniteDSpritesAnalogies, InfiniteDSpritesTriplets],
)
def test_idsprites_instantiation_with_no_parameters(dataset_class):
    """Test that the dataset can be instantiated with no parameters."""
    dataset = dataset_class()
    assert dataset.img_size == 256
    assert dataset.ranges["color"] == ["white"]
    assert np.allclose(dataset.ranges["scale"], np.linspace(0.5, 1.0, 32))
    assert np.allclose(
        dataset.ranges["orientation"], np.linspace(0, 2 * np.pi * 32 / 33, 32)
    )
    assert np.allclose(dataset.ranges["position_x"], np.linspace(0, 1, 32))
    assert np.allclose(dataset.ranges["position_y"], np.linspace(0, 1, 32))


@pytest.mark.parametrize(
    "dataset_class",
    [InfiniteDSprites, InfiniteDSpritesAnalogies, InfiniteDSpritesTriplets],
)
def test_instantiation_from_config(dataset_class):
    """Test that the dataset can be instantiated from a config."""
    config = {
        "img_size": 64,
        "color_range": ["red", "green", "blue"],
        "scale_range": [0.5, 1.0],
        "orientation_range": {
            "start": 0.0,
            "stop": 2 * np.pi,
            "num": 10,
        },
        "position_x_range": np.linspace(0.1, 0.9, 3),
        "position_y_range": np.linspace(0.1, 0.9, 3),
    }
    dataset = dataset_class.from_config(config)
    assert dataset.img_size == 64
    assert dataset.ranges["color"] == ["red", "green", "blue"]
    assert dataset.ranges["scale"] == [0.5, 1.0]
    assert np.allclose(dataset.ranges["orientation"], np.linspace(0.0, 2 * np.pi, 10))
    assert np.allclose(dataset.ranges["position_x"], np.linspace(0.1, 0.9, 3))
    assert np.allclose(dataset.ranges["position_y"], np.linspace(0.1, 0.9, 3))


@pytest.mark.parametrize(
    "img_size,color_range,grayscale,expected_shape",
    [(244, ["white"], True, (1, 244, 244)), (256, ["red"], False, (3, 256, 256))],
)
def test_idsprites_image(img_size, color_range, grayscale, expected_shape):
    """Test that the dataset can be iterated over and the image has the expected dimmensions."""
    dataset = InfiniteDSprites(
        img_size=img_size, color_range=color_range, grayscale=grayscale
    )
    image, _ = next(iter(dataset))
    assert image.shape == expected_shape
    assert image.min() >= 0.0
    assert image.max() <= 1.0


@pytest.mark.parametrize(
    "img_size,color_range,expected_shape",
    [(244, ["white"], (3, 244, 244)), (256, ["red"], (3, 256, 256))],
)
def test_idsprites_triplets_image(img_size, color_range, expected_shape):
    """Test that the dataset can be iterated over."""
    dataset = InfiniteDSpritesTriplets(img_size=img_size, color_range=color_range)
    images, _ = next(iter(dataset))
    assert images[0].shape == expected_shape
    assert images[0].min() >= 0.0
    assert images[0].max() <= 1.0


@pytest.mark.parametrize(
    "img_size,color_range,expected_shape",
    [(244, ["white"], (3, 244, 244)), (256, ["red"], (3, 256, 256))],
)
def test_idsprites_analogies_image(img_size, color_range, expected_shape):
    """Test that the dataset can be iterated over."""
    dataset = InfiniteDSpritesAnalogies(img_size=img_size, color_range=color_range)
    image = next(iter(dataset))
    assert image.shape == expected_shape
    assert image.min() >= 0.0
    assert image.max() <= 1.0


@pytest.mark.parametrize(
    "dataset_class",
    [InfiniteDSprites, InfiniteDSpritesAnalogies, InfiniteDSpritesTriplets],
)
@pytest.mark.parametrize("size", [-1, 0, 1, 5])
def test_dataset_size(dataset_class, size):
    """Test that setting the dataset size works."""
    dataset = dataset_class(dataset_size=size)
    data = list(dataset)
    assert len(data) == max(0, size)


@pytest.mark.parametrize("shapes", [-1, 0, 1, 5])
def test_dataset_size_shapes(shapes):
    """Test that setting the maximum number of shapes works."""
    dataset = InfiniteDSprites(
        scale_range=[0.5, 1.0],
        orientation_range=[0.0, 1.0],
        position_x_range=[0.0, 1.0],
        position_y_range=[0.0, 1.0],
        shapes=shapes,
    )
    data = list(dataset)
    assert len(data) == max(0, shapes) * np.prod(
        [len(r) for r in dataset.ranges.values()]
    )
