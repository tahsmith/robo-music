import utils
import numpy as np


def test_dilate_zero_order_hold():
    x = np.array([[1, 1], [2, 2], [3, 3]])
    expected = [[1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [3, 3], [3, 3],
                [3, 3]]
    acutal = utils.upsample_zero_order_hold(x, 3)

    assert (expected == acutal).all()