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
from typing import Any, Dict, Tuple, Type

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
    def render(self, axes: matplotlib.axes.Axes, **kwargs) -> None:
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
        self.render(axes, **kwargs)
        return axes


class TypeProxy:

    """Base class for type proxy classes.

    This class implements a simple type proxy mechanism, allowing to register
    different dataclass types under different names, and to create concrete
    objects of the desired type based on keyword arguments.

    .. warning::

       Only dataclass types can be registered. This is intrumental for the
       inner workings of the type proxy mechanism, since we rely on the
       dataclass fields to filter the keyword arguments when creating concrete objects.

    Arguments
    ---------
    key : str
        The name of the keyword argument that will be used to select the desired type
        when creating objects from keyword arguments.
    """

    def __init__(self, key: str) -> None:
        """Constructor.
        """
        self._key = key
        self._proxy_dict: Dict[str, Type[Any]] = {}
        self._default = None

    def key(self) -> str:
        """Return the key used to select the desired type.

        This is useful, e.g., for argparse argument names.

        Returns
        -------
        str
            The key used to select the desired type.
        """
        return self._key

    def register(self, name: str, cls: type, default: bool = False) -> None:
        """Register a new dataclass in the proxy dictionary.

        Arguments
        ---------
        name : str
            The name of the type to register.

        cls : class
            The dataclass type to register.
        """
        # Note that is_dataclass() returns True on both dataclass types and instances.
        if not (isinstance(cls, type) and is_dataclass(cls)):
            raise TypeError(f"{cls} is not a dataclass type and cannot be registered.")
        if name in self._proxy_dict:
            raise ValueError(f"{name!r} already registered.")
        self._proxy_dict[name] = cls
        if default or self._default is None:
            self._default = name

    def choices(self) -> Tuple[str, ...]:
        """Return the available choices.

        This is useful for argparse choices, for instance.

        Returns
        -------
        tuple of str
            The available choices.
        """
        return tuple(self._proxy_dict.keys())

    def default(self) -> str:
        """Return the default choice.

        This is useful for argparse default values, for instance.

        Returns
        -------
        str
            The default choice.
        """
        if self._default is None:
            raise RuntimeError(f"No default type registered in {self.__class__.__name__}.")
        return self._default

    def _cls(self, name: str) -> Type[Any]:
        """Return the type corresponding to the given name.

        Arguments
        ---------
        name : str
            The name of the desired type.

        Returns
        -------
        class
            The class of the desired type.
        """
        if name not in self._proxy_dict:
            raise ValueError(f"Unknown proxy type {name!r} for {self.__class__.__name__}.")
        return self._proxy_dict[name]

    def create(self, name: str,**kwargs) -> Any:
        """Create an object of the desired type.

        Arguments
        ---------
        name : str
            The name of the desired type.

        kwargs : keyword arguments
            The keyword arguments to pass to the constructor.

        Returns
        -------
        object
            The created object.
        """
        cls = self._cls(name)
        return cls(**kwargs)

    @staticmethod
    def filter_dataclass_kwargs(cls: type, kwargs: dict) -> dict:
        """Filter keyword arguments to keep only those defined in the dataclass.

        Arguments
        ---------
        cls : class
            The dataclass type.

        kwargs : dict
            The keyword arguments to filter.

        Returns
        -------
        dict
            The filtered keyword arguments.
        """
        return {key: value for key, value in kwargs.items() if key in cls.__dataclass_fields__}

    def from_kwargs(self, **kwargs) -> Any:
        """Create an object of the desired type based only on keyword arguments.

        This is extracting the desired type from the keyword arguments using
        the configured key, and filtering the keyword arguments to keep only those
        defined in the target dataclass.

        .. warning::

           Use this with caution, as we are silently dropping any keyword argument
           that is not defined in the target dataclass. This may be fine when the
           class acts as an interface layer to argparse, but if you use the function
           in the wild, you might end up having typos in your keyword arguments that
           go undetected.

        Arguments
        ---------
        kwargs : keyword arguments
            The keyword arguments to pass to the constructor.

        Returns
        -------
        object
            The created object.
        """
        name = kwargs.get(self._key) or self.default()
        cls = self._cls(name)
        kwargs = self.filter_dataclass_kwargs(cls, kwargs)
        return cls(**kwargs)
