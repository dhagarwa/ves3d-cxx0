#!/usr/bin/env python
'''plot the shapes stored in the input file'''

from __future__ import absolute_import, division, print_function

__author__    = 'Abtin Rahimian'
__email__     = 'arahimian@acm.org'
__status__    = 'prototype'
__revision__  = '$Revision$'
__date__      = '$Date$'
__tags__      = '$Tags$'
__copyright__ = 'Copyright (c) 2015, Abtin Rahimian'
__license__   = '''
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
'''

import argparse as ap
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

lgr = logging.getLogger(__name__)

def parse_args():
    # one can use click, or cliff for better control
    p = ap.ArgumentParser(description='Loads and plots the shape dumped by ves3d code.')
    p.add_argument('-p', help='spherical harmonics order', type=int)
    p.add_argument('-n', help='number of surfaces', default=1, type=int)
    p.add_argument('--sub-index','-s', nargs='*', help='surface index', default=None, type=int)
    p.add_argument('--out-template', '-o', help='output file name', default=None)
    p.add_argument('--animate', '-a', help='animate the plot', action='store_true')
    p.add_argument('--increment', '-i', help='animate the plot', default=1, type=int)
    p.add_argument('files', nargs=ap.REMAINDER, help='list of files')

    args = p.parse_args()
    return vars(args)

def load_file(fnames):

    data = list()
    for f in fnames:
        with open(f, 'r+') as fh:
            for line in fh:
                if (line[:6]=='data: '):
                    X=line[6:].strip().split(' ')
                    data.append(np.array([float(x) for x in X]))

    data = np.hstack(data)
    return data

def plot_series( data, p, n, out_template, animate,
                 increment, **kwargs):

    sz   = 2*p*(p+1)
    data = data.reshape((3*sz*n,-1),order='F')
    nT   = data.shape[1]

    fig = plt.figure()
    ax  = plt.gca(projection='3d')
    nS  = range(n) if kwargs['sub_index'] is None else kwargs['sub_index']

    for iT in range(0,nT,increment):
        print('step %d/%d' % (iT, nT))
        ax.cla()
        xyz = data[:,iT].reshape((3*sz,-1), order="F")

        for iN in nS:
            plot(p=p,
                 ax=ax,
                 x = xyz[   0:  sz,iN],
                 y = xyz[  sz:2*sz,iN],
                 z = xyz[2*sz:    ,iN])

        if animate:
            el=20-8*(iT/nT/2-1)**2
            az=120+160*iT/nT
        else:
            el=0
            az=-90

        ax.view_init(elev=el,azim=az)

        cen = np.mean(xyz.reshape((3,-1),order='C'),axis=1,keepdims=True)
        rad = np.amax(np.abs(xyz.reshape((3,-1),order='C')-cen),axis=1)

        # ax.set_xlim3d(cen[0]-rad[0],cen[0]+rad[0])
        # ax.set_ylim3d(cen[1]-rad[1],cen[1]+rad[1])
        # ax.set_zlim3d(cen[2]-rad[2],cen[2]+rad[2])

        # l=1-.85*(iT/nT)
        # ax.set_xlim3d(cen[0]-l,cen[0]+l)
        # ax.set_ylim3d(cen[2]-l,cen[2]+l)
        # ax.set_zlim3d(cen[2]-l,cen[2]+l)
        ax.set_aspect('equal')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_axis_off()
        ax.w_xaxis.line.set_visible(False)
        ax.w_yaxis.line.set_visible(False)
        ax.w_zaxis.line.set_visible(False)

        plt.draw()
        plt.pause(.2)

        if out_template:
            fname = out_template % iT
            plt.savefig(fname, transparent=True,dpi=100)

    plt.show()

def plot(p,ax,x,y,z):

    x = x.reshape((p+1,-1))
    y = y.reshape((p+1,-1))
    z = z.reshape((p+1,-1))

    x = np.hstack((x, x[:,0,None]))
    y = np.hstack((y, y[:,0,None]))
    z = np.hstack((z, z[:,0,None]))

    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, color='r',
                           linewidth=.1, antialiased=False,alpha=.8)

def main():
    opts = parse_args()
    data = load_file(opts['files'])
    plot_series(data=data,**opts)

if __name__ == '__main__':
    main()
