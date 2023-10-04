# Copyright (C) 2023 luca.baldini@pi.infn.it
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

"""Pretty printing.
"""

from enum import Enum


class AnsiFontEffect(Enum):

    """Small enum class to support colors and advanced formatting in the
    rendering of digitized event data.

    See https://stackoverflow.com/questions/4842424/list-of-ansi-color-escape-sequences
    for a useful recap of the basic rules.
    """

    RESET = 0
    BOLD = 1
    UNDERLINE = 4
    FG_BLACK = 30
    FG_RED = 31
    FG_GREEN = 32
    FG_YELLOW = 33
    FG_BLUE = 34
    FG_MAGENTA = 35
    FG_CYAN = 36
    FG_WHITE = 37
    BG_BLACK = 40
    BG_RED = 41
    BG_GREEN = 42
    BG_YELLOW = 43
    BG_BLUE = 44
    BG_MAGENTA = 45
    BG_CYAN = 46
    BG_WHITE = 47
    FG_BRIGHT_BLACK = 90
    FG_BRIGHT_RED = 91
    FG_BRIGHT_GREEN = 92
    FG_BRIGHT_YELLOW = 93
    FG_BRIGHT_BLUE = 94
    FG_BRIGHT_MAGENTA = 95
    FG_BRIGHT_CYAN = 96
    FG_BRIGHT_WHITE = 97
    BG_BRIGHT_BLACK = 100
    BG_BRIGHT_RED = 101
    BG_BRIGHT_GREEN = 102
    BG_BRIGHT_YELLOW = 103
    BG_BRIGHT_BLUE = 104
    BG_BRIGHT_MAGENTA = 105
    BG_BRIGHT_CYAN = 106
    BG_BRIGHT_WHITE = 107



def ansi_format(text : str, *effects : AnsiFontEffect) -> str:
    """Return the proper combination of escape sequences to render
    a given piece of text with a series of font effects.

    Arguments
    ---------
    text : str
        The text to be rendered with special effects.

    effects : AnsiFontEffect
        The effects to be applied to the text.
    """
    esc = ';'.join([f'{effect.value}' for effect in effects])
    return f'\033[{esc}m{text}\033[{AnsiFontEffect.RESET.value}m'

def _repeat(text : str, repetitions : int) -> str:
    """Repeat a given piece of text for a given number of times.
    """
    return text * repetitions

def space(width : int) -> str:
    """Return a sequence of spaces of a given width.
    """
    return _repeat(' ', width)

def line(width : int) -> str:
    """Return a sequence of spaces of a given width.
    """
    return _repeat('-', width)
