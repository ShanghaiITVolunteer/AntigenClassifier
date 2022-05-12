# -*- coding: utf-8 -*-
# __author__:Livingbody
# 2022/5/7 0:04
import pytest
from ../src/masking/data_mask import data_mask


def data_mask_test():
    """ doc """
    data_mask(source_dir="antigen-images", target_dir="saved")


if __name__ == "__main__":
    data_mask_test()
