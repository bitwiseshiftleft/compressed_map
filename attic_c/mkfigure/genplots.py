#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log,ceil
from matplotlib import pyplot as plt
import matplotlib.font_manager as mfm
from matplotlib.font_manager import FontProperties

import sys
sys.path.append(r"results_py")
from fringe_opz_orig_beta2 import data as results_fringe_opz2
from fringe_opz_orig_beta4 import data as results_fringe_opz4
from fringe_opw_beta2 import data as results_fringe_opw2
from fringe_opv_beta2 import data as results_fringe_opv2
from fringe_opv_beta4 import data as results_fringe_opv4
from fringe_opv_beta8 import data as results_fringe_opv8
from hier_step121_beta4   import data as results_hier
from hier_shr_beta4   import data as results_hier_shr
from hier_opv_beta4   import data as results_hier_op0_4
from hier_opv_beta8   import data as results_hier_op0_8
from hier_shr_beta8   import data as results_hier_shr8
from fringe_shr_beta4   import data as results_fringe_shr
from fringe_shr_beta8   import data as results_fringe_shr8
from hier_step121_beta8   import data as results_hier8
from fringe_step121_beta4 import data as results_fringe
from fringe_step121_beta8 import data as results_fringe8
#from fringe_step121_pp2_beta4 import data as results_fringe_pp
#from fringe_step121_p2_beta4  import data as results_fringe_p
from fringe_step131_p2_beta4  import data as results_fringe_p13
from fringe_step131_p2_beta8  import data as results_fringe_p13_8
from fringe_step121_pp2_beta8 import data as results_fringe_pp8
from fringe_step161_3lg_beta4 import data as results_fringe_lg

data = [
    #(results_fringe,     u"frayed (β=32)"),
    # (results_fringe8,    u"frayed (β=64)"),
    # (results_fringe_pp,  u"frayed (β=32, pp)"),
    #(results_fringe_p,   u"frayed (β=32, p)"),
    #(results_fringe_p13, u"frayed (β=32)"), # p13
    #(results_fringe_opz2, u"frayed (β=16 op 0.1%)"),
    #(results_fringe_opw2, u"frayed (β=16), opw"),
    (results_fringe_opv2, u"frayed (β=16, ε=0.1%)"),
    (results_fringe_opv4, u"frayed (β=32, ε=0.1%)"),
    (results_fringe_opv8, u"frayed (β=64, ε=0.1%)"),
    #(results_fringe_shr, u"frayed (β=32)"), # p13
    #(results_fringe_shr8, u"frayed (β=64)"), # p13
    #(results_fringe_pp8, u"frayed (β=64, pp)"),
    #(results_fringe_p13_8, u"frayed (β=64)"), # p13
   #    (results_fringe_shr8, u"frayed (β=64)"), # p13
    # (results_fringe_lg,  u"frayed (β=32, lg)"),
    #(results_hier,       u"Calderesque (β=32)"),
    #(results_hier_shr,       u"Calderesque (β=32)"),
    (results_hier_op0_4,       u"Calderesque (β=32)"),
    (results_hier_op0_8,      u"Calderesque (β=64)"),
    #(results_hier8,      u"Calderesque (β=64)"),
]

for d,_ in data: d.sort()

plt.subplots_adjust(left=0,right=0.9,top=0.9)

def sparsify(r, delta=2**0.1):
    out = []
    prev = None
    for row in r:
        if prev is None or row[0] >= prev*delta or row is r[-1]:
            out.append(row)
            prev = row[0]
    return out

data = [(sparsify(d),n) for d,n in data]

prop = mfm.FontProperties(family="Arial", size=12, weight="normal")

fig,ax = plt.subplots(figsize=(6,4.5))
ax.set_xscale("log",basex=10)
#ax.set_xlim([64, 1<<int(ceil(max(log(n,2) for r in [results_hier,results_fringe] for (n,p,r,b,q) in r)))])
ax.set_xlabel("rows")
ax.set_yscale("log")
ax.set_ylabel(u"solution time (seconds)")
#ax.plot([n for (n,p,r,b,q) in results_fringe_adj_sp], [t/p for (n,p,r,b,q) in results_fringe_adj_sp], label=u"frayed adj")
for d,l in data:
    ax.plot([n for (n,p,s,r,b,q) in d], [n*b/1e6 for (n,p,s,r,b,q) in d], label=l)
ax.legend(loc="upper left", prop=prop)
fig.savefig("solution_time.png", dpi=600)

fig,ax = plt.subplots(figsize=(6,4.5))
ax.set_xscale("log",basex=10)
#ax.set_xlim([64, 1<<int(ceil(max(log(n,2) for r in [results_hier,results_fringe] for (n,p,s,r,b,q) in r)))])
ax.set_xlabel("rows")
ax.set_yscale("log",basey=10)
ax.set_ylabel(u"solution time (s / row)")
ax.set_ylim([10**-7,10**-5])
#ax.plot([n for (n,p,r,b,q) in results_fringe_adj_sp], [t*1e6/(p*n) for (n,p,r,b,q) in results_fringe_adj_sp], label=u"frayed adj")
for d,l in data:
    ax.plot([n for (n,p,s,r,b,q) in d], [(b+s)/1.e6 for (n,p,s,r,b,q) in d], label=l)
ax.legend(loc="upper left", prop=prop)
fig.savefig("usec_per_row.png", dpi=600)

fig,ax = plt.subplots(figsize=(6,4.5))
ax.set_xscale("log",basex=10)
#ax.set_xlim([64, 1<<int(ceil(max(log(n,2) for r in [results_hier,results_fringe] for (n,p,s,r,b,q) in r)))])
ax.set_ylim([0,150])
ax.set_xlabel("rows")
ax.set_ylabel(u"query time (ns / row)")
for d,l in data:
    ax.plot([n for (n,p,s,r,b,q) in d], [q*1000 for (n,p,s,r,b,q) in d], label=l)
ax.legend()
ax.legend(loc="upper left", prop=prop)
fig.savefig("query_speed.png", dpi=600)

fig,ax = plt.subplots(figsize=(6,4.5))
ax.set_xscale("log",basex=10)
#ax.set_xlim([64, 1<<int(ceil(max(log(n,2) for r in [results_hier,results_fringe] for (n,p,r,b,q) in r)))])
ax.set_xlabel("rows")
ax.set_ylabel(u"success rate (%)")
for d,l in data:
    ax.plot([n for (n,p,s,r,b,q) in d], [100*p for (n,p,s,r,b,q) in d], label=l)
ax.legend(loc="lower left", prop=prop)
fig.savefig("success_rate.png", dpi=600)