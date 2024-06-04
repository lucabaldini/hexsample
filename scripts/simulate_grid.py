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

"""Simulate a grid of configurations.
"""
import numpy as np

from hexsample import HEXSAMPLE_DATA, logger
from hexsample.pipeline import hxsim, hxrecon


# Number of events to be generated for each configuration.
NUM_EVENTS = 100000
# Detector thickness grid in cm.
THICKNESS = (0.02, 0.025, 0.03, 0.035)
# Equivalent noise charge grid in e.
NOISE = (20, 30, 40)
# Chip pitch in cm
PITCH = (0.0050, 0.0055, 0.0060)
# Number of neirest neighbors in track clusters
NUM_NEIGHBORS = np.arange(1,8)
# Zero-suppression threshold, expressed in units of enc.
SIGMA_THRESHOLD = np.linspace(0.5, 4, 30)
# Number of neighbors for the clustering.
NUM_NEIGHBORS = 2

'''
for thickness in THICKNESS:
    for noise in NOISE:
        for pitch in PITCH:
            # Simulate...
            file_name = f'sim_{1.e4 * thickness:.0f}um_{noise:.0f}enc_{1e4 * pitch:.0f}pitch.h5'
            file_path = HEXSAMPLE_DATA / file_name
            kwargs = dict(outfile=file_path, thickness=thickness, noise=noise, pitch=pitch)
            file_path = hxsim(numevents=NUM_EVENTS, **kwargs)
            # ... and reconstruct.
            threshold = noise * SIGMA_THRESHOLD
            suffix = f'recon_nn{NUM_NEIGHBORS}_thr{threshold:.0f}'
            kwargs = dict(zsupthreshold=threshold, nneighbors=NUM_NEIGHBORS, suffix=suffix)
            hxrecon(infile=file_path, **kwargs)
'''

for thr in SIGMA_THRESHOLD:
    for nn in NUM_NEIGHBORS:
        # Simulate...
            file_name = f'sim_{thr:.0f}enc_20enc_{1e4 * pitch:.0f}pitch.h5'
            file_path = HEXSAMPLE_DATA / file_name
            kwargs = dict(outfile=file_path, thickness=0.025, noise=20, pitch=0.006)
            file_path = hxsim(numevents=NUM_EVENTS, **kwargs)
            # ... and reconstruct.
            threshold = 20 * SIGMA_THRESHOLD
            suffix = f'recon_nn{NUM_NEIGHBORS}_thr{threshold:.0f}'
            kwargs = dict(zsupthreshold=threshold, nneighbors=NUM_NEIGHBORS, suffix=suffix)
            hxrecon(infile=file_path, **kwargs)


