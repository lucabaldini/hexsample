# Copyright (C) 2024 the hexample team.
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

"""Create a simple simulation with a sparse readout for execising the test bench.
"""

from loguru import logger
from tqdm import tqdm

from hexsample import rng, HEXSAMPLE_DATA
from hexsample.readout import HexagonalReadoutSparse
from hexsample.hexagon import HexagonalLayout
from hexsample.mc import PhotonList
from hexsample.source import LineForest, GaussianBeam, Source
from hexsample.sensor import Material, Sensor


output_file_path = HEXSAMPLE_DATA / 'test_bench_cu.txt'

kwargs = dict(seed=None, srcelement='Cu', srclevel='K', srcposx=0., srcposy=0.,
    srcsigma=0.02, actmedium='Si', fano=0.116, thickness=0.03, transdiffsigma=40.,
    numevents=1000, layout='ODD_R', numcolumns=32, numrows=32, pitch=0.005, noise=0.,
    gain=1., trgthreshold=200., zsupthreshold=0., offset=0)


rng.initialize(seed=kwargs['seed'])
spectrum = LineForest(kwargs['srcelement'], kwargs['srclevel'])
beam = GaussianBeam(kwargs['srcposx'], kwargs['srcposy'], kwargs['srcsigma'])
source = Source(spectrum, beam)
material = Material(kwargs['actmedium'], kwargs['fano'])
sensor = Sensor(material, kwargs['thickness'], kwargs['transdiffsigma'])
photon_list = PhotonList(source, sensor, kwargs['numevents'])
args = HexagonalLayout(kwargs['layout']), kwargs['numcolumns'], kwargs['numrows'],\
     kwargs['pitch'], kwargs['noise'], kwargs['gain']
readout = HexagonalReadoutSparse(*args)
logger.info(f'Readout chip: {readout}')
readout_args = kwargs['trgthreshold'], kwargs['zsupthreshold'], kwargs['offset']
logger.info(f'Opening output file {output_file_path}...')
output_file = open(output_file_path, 'w')
logger.info('Starting the event loop...')
for mc_event in tqdm(photon_list):
    x, y = mc_event.propagate(sensor.trans_diffusion_sigma)
    event = readout.read(mc_event.timestamp, x, y, *readout_args)
    output_file.write(f'{event.trigger_id}    {event.timestamp():.9f}    {len(event.pha)}\n')
    for col, row, pha in zip(event.columns, event.rows, event.pha):
        output_file.write(f'{col:2}    {row:2}    {pha}\n')
output_file.close()
logger.info('Output file closed.')
