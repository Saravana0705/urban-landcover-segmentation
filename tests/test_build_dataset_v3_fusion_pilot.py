import numpy as np

from src.data.build_dataset_v3_fusion_pilot import IGNORE_INDEX, fuse_block


def test_conservative_fusion_and_precedence():
    osm = np.array([[1, 2, 5, 4, 3, 0, 4, 0]], dtype=np.uint8)
    dw = np.array([[3, 3, 5, 0, 3, 4, 4, 0]], dtype=np.uint8)
    ua = np.array([[3, 3, 5, 4, 3, 4, 0, 0]], dtype=np.uint8)
    codes = np.array([[21000, 21000, 50000, 13100, 21000, 13100, 13400, 0]], dtype=np.uint16)
    semantic, validity, provenance, support, conflict = fuse_block(osm, dw, ua, codes)
    assert semantic.tolist() == [[1, 2, 5, 4, 3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]]
    assert validity.tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]
    assert provenance.tolist() == [[1, 1, 7, 5, 7, 0, 0, 0]]
    assert support.tolist() == [[1, 1, 3, 2, 3, 0, 0, 0]]
    assert conflict[0, 0] == 1 and conflict[0, 1] == 1


def test_no_default_vegetation():
    zeros = np.zeros((2, 3), dtype=np.uint8)
    semantic, validity, *_ = fuse_block(zeros, zeros, zeros, np.zeros((2, 3), dtype=np.uint16))
    assert np.all(semantic == IGNORE_INDEX)
    assert not validity.any()
