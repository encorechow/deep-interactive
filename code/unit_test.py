import unittest
import numpy as np
from sample_clicks import *
from cfg.config import cfg

class TestSampleClicks(unittest.TestCase):
    def test_load_image_path(self):
        images, _ = load_image_path(cfg.IMG_DIR, cfg.IMG_EXT)
        gts = load_image_path(cfg.GT_DIR, cfg.GT_EXT)

    def test_pos_samples(self):
        gts, _ = load_image_path(cfg.GT_DIR, cfg.GT_EXT)
        cat_channels(gts[0])
if __name__ == '__main__':
    unittest.main()
