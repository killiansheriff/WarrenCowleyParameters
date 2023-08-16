import numpy as np
import pytest
from ovito.data import DataCollection
from ovito.io import import_file
from ovito.modifiers import ExpressionSelectionModifier

from WarrenCowleyParameters import WarrenCowleyParameters


@pytest.fixture
def import_data():
    pipe = import_file("examples/fcc.dump")
    yield pipe.compute()


def test_default_settings(import_data: DataCollection):
    data = import_data
    data.apply(WarrenCowleyParameters())

    expected_wc_1NN = np.array(
        [
            [0.33578938, -0.24112737, -0.09491391],
            [-0.24112737, 0.06615814, 0.17515013],
            [-0.09491391, 0.17515013, -0.08016501],
        ]
    )

    wc_for_shells = data.attributes["Warren-Cowley parameters"]

    assert np.allclose(expected_wc_1NN, wc_for_shells[0])


def test_wc_2NN(import_data: DataCollection):
    data = import_data
    data.apply(WarrenCowleyParameters(nneigh=[0, 12, 18]))
    expected_wc_2NN = np.array(
        [
            [-0.40634331, 0.21044729, 0.19620085],
            [0.21044729, -0.11655821, -0.09404696],
            [0.19620085, -0.09404696, -0.10230108],
        ]
    )
    wc_for_shells = data.attributes["Warren-Cowley parameters"]
    assert np.allclose(expected_wc_2NN, wc_for_shells[1])


def test_wc_symmetric(import_data: DataCollection):
    data = import_data
    data.apply(WarrenCowleyParameters())
    wc_for_shells = data.attributes["Warren-Cowley parameters"]
    wc = wc_for_shells[0]
    assert np.allclose(wc, wc.T)


def test_selection(import_data: DataCollection):
    # Not implemented yet
    # data = import_data
    # data.apply(ExpressionSelectionModifier(expression=""))
    # data.apply(WarrenCowleyParameters())

    # wc_for_shells = data.attributes["Warren-Cowley parameters"]
    # wc = wc_for_shells[0]

    # expected_wc = np.array([])

    # assert np.allclose(expected_wc, wc)
    pass


def test_wc_shape(import_data: DataCollection):
    data = import_data
    data.apply(WarrenCowleyParameters())

    wc_for_shells = data.attributes["Warren-Cowley parameters"]
    wc = wc_for_shells[0]

    assert wc.shape == (3, 3)
