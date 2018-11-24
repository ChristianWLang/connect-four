"""
Base class for talos environments. If you wish to build an environment, it should inherit this class. If you choose to build an environment and choose not to inherit this class, then make sure to either include these attributes in your environment, or edit the algorithm to reflect the changes in attributes.
"""
# Author: Christian Lang <me@christianlang.io>

import numpy as np


class Base(object):
    def __init__(self):
        self.state = None
        pass

    def get_state(self):
        return np.copy(self.state)
