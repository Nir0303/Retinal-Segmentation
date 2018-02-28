# /usr/bin/env python
"""
utility module to run os commands
"""
from __future__ import print_function

import os
import shutil

def create_directory(path):
    """
    create directory if not exists
    :param path: path of directory
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)


def remove_directory(path):
    """
    remove directory if exists
    :param path: path of directory
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)

