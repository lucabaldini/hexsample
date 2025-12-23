# Copyright (C) 2023--2025 the hexsample team.
#
# For the license terms see the file LICENSE, distributed along with this
# software.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Base classes and functions shared across modules.
"""

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import Tuple

import matplotlib
import numpy as np
from aptapy.plotting import plt


class AbstractRandomGenerator(ABC):

    """Abstract base class for random number generators.
    """

    @abstractmethod
    def rvs(self, size: int = 1) -> Tuple[np.ndarray, ...]:
        """Generate random variates.

        Arguments
        ---------
        size : int
            The number of random variates to be generated.

        Returns
        -------
        values : (tuple of) np.ndarray of shape `size`
            The actual random variates.
        """


class AbstractPlottable(ABC):

    """Abstract base class for plottable objects.
    """

    @abstractmethod
    def _render(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
        """Render the object on the given axes.

        Arguments
        ---------
        axes : matplotlib.axes.Axes
            The axes to plot on.

        kwargs : keyword arguments
            Additional keyword arguments passed to the rendering method.
            Note that the specifics depends on how _render() is implemented, and
            which type of matplotlib object the plottable is representing.
        """

    def plot(self, axes: matplotlib.axes.Axes = None, **kwargs) -> matplotlib.axes.Axes:
        """Plot the object on the given axes (or on the current axes if none
        is passed as an argument).

        Arguments
        ---------
        axes : matplotlib.axes.Axes, optional
            The axes to plot on. If None, the current axes are used.

        kwargs : keyword arguments
            Additional keyword arguments passed to the _render() method.
            Note that the specifics depends on how _render() is implemented, and
            which type of matplotlib object the plottable is representing.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the object has been plotted on.
        """
        if axes is None:
            axes = plt.gca()
        self._render(axes, **kwargs)
        return axes


def type_proxy(target_class: type = None, *, default: str = None):

    """Dual-mode class decorator to create type proxy classes.

    Since we have a few situations in which we might instantiate different type
    of objects depending on user input (e.g., different spectrum types, or
    different beam shapes), we provide this decorator to create type proxy classes
    that can be used to manage such situations in a uniform way, dispatching things
    accordingly.

    The class to be decorated must define two class attributes:
    - `_KEY`: the name of the keyword argument that will be used to select the desired type;
    - `_PROXY_DICT`: a dictionary mapping object names (strings) to dataclass types.
    Everything else happens more or less automatically.

    Arguments
    ----------
    target_class: class, optional
        The target class to decorate. If not provided, the decorator is assumed
        to be used with arguments.

    default: str, optional
        The default object name to use when none is provided. If not provided, the first
        object in the proxy dictionary is used.
    """

    _REQUIRED_ATTRS = ("_KEY", "_PROXY_DICT")

    def wrapper(cls):
        """Class decorator implementation.
        """
        # Validate the target class: it must define _KEY and _PROXY_DICT class attributes.
        for attr in _REQUIRED_ATTRS:
            if not hasattr(cls, attr):
                raise ValueError(f"Type-proxy classes must define a {attr} attribute.")

        # Validate the proxy dictionary: all values must be dataclass types.
        for type_ in cls._PROXY_DICT.values():
            if not is_dataclass(type_):
                raise ValueError(f"{type_} is not a dataclass.")

        # Determine default key
        default_name = default or next(iter(cls._PROXY_DICT))

        # Add static methods to retrieve default and available choices.
        cls.key = staticmethod(lambda: cls._KEY)
        cls.default = staticmethod(lambda: default_name)
        cls.choices = staticmethod(lambda: tuple(cls._PROXY_DICT.keys()))

        # Add factory method to create objects of the desired type.
        def factory(cls, **kwargs):
            """Factory method to create objects of the desired type.

            Note this will filter out silently any unexpected keyword arguments
            not defined in the target dataclass. This is ok when used in
            conjunction with argument parsers, where the input is controlled, but
            is error-prone when used in the wild.
            """
            object_name = kwargs.get(cls._KEY, default_name)
            if object_name not in cls._PROXY_DICT:
                raise ValueError(f"Unknown {cls._KEY} type: {object_name!r}")
            cls_ = cls._PROXY_DICT[object_name]
            kwargs = {k: v for k, v in kwargs.items() if k in cls_.__dataclass_fields__}
            return cls_(**kwargs)

        cls.factory = classmethod(factory)

        return cls

    # Proxy used without arguments, i.e., @type_proxy
    if target_class is not None:
        return wrapper(target_class)

    # Proxy used with arguments, i.e., @type_proxy(default=)
    return wrapper
