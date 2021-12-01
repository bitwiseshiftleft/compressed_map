#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log,ceil,floor
from matplotlib import pyplot as plt
import matplotlib.font_manager as mfm
from matplotlib.font_manager import FontProperties

import sys
sys.path.append(r"results_py")
from sketch_fringe   import data as results_fringe
from sketch_hier     import data as results_hier

def crlite(p,f=log(2.)):
    # TODO: integrality
    if p<=0: return 0
    return p * (log(1/p,2.)/f + 4.2*log(2.)/f)

def ent(p):
    if p<=0 or p>=1: return 0
    return (p*log(p) + (1-p)*log(1-p)) / log(0.5)

def lp_inspired_opt(prs,verbose=False):
    EPSILON = 0.01
    prs = list(sorted(prs))
    
    def pot(x): return 2**floor(log(x*1.,2.))
    def pod(x): return 2**(ceil(log(x*1.,2.)-EPSILON)-1)
    def bits_to_express(p):
        c = floor(log(p,0.5))
        p *= 2**c
        return c + (2-2*p)
    
    qs = [pot(pr) for pr in prs]
    qs[-1] = 1-sum(qs[:-1])
    def deriv(i): return prs[i]/pot(qs[i]) - prs[-1]/pod(qs[-1])
    for _ in range(100): # to prevent it from going off the rails
        dm,i = max((deriv(i),i) for i in range(len(qs)-1))
        #print(qs,dm,i)
        if dm <= 0: break
        amt = min(2*pot(qs[i])-qs[i], qs[-1]-pod(qs[-1]))
        qs[i] += amt
        assert(qs[i] <= 1)
        qs[-1] -= amt
        
    return sum(p*bits_to_express(q) for p,q in zip(prs,qs))
def new_est(p): return lp_inspired_opt([p,1-p])

xcoords = [x/100000. for x in range(1,50001)]

prop = mfm.FontProperties(family="Arial", size=12, weight="normal")

fig,ax = plt.subplots(figsize=(6,4.5))
ax.set_xlim(1e-5,0.5)
ax.set_xlabel("fraction revoked")
ax.set_ylabel(u"bits per revocation")
ax.set_xscale("log")
#ax.set_yscale("log")
ax.plot(xcoords, [crlite(x)/x for x in xcoords], label="CRLite (est.)")
#ax.plot(xcoords, [crlite(x,1) for x in xcoords], label="CRLite/matrix (est.)")
ax.plot([min(p,1-p) for (p,t,b,q) in results_fringe], [b*8/1.e7/p for (p,t,b,q) in results_fringe], label="Ours (measured, 10M certs)")
ax.plot(xcoords, [new_est(x)/x for x in xcoords], label="Ours (asymptotic)")
ax.plot(xcoords, [ent(x)/x for x in xcoords], label="Entropy limit")
ax.legend(loc="upper right", prop=prop)
fig.savefig("entropy.png", dpi=600)


fig,ax = plt.subplots(figsize=(6,4.5))
ax.set_xlim(1e-4,0.5)
ax.set_xlabel("fraction revoked")
ax.set_ylabel(u"size / entropy")
ax.set_xscale("log")
# ax.set_ylim(0,3)
#ax.set_yscale("log")
#ax.plot(xcoords, [crlite(x)/ent(x) for x in xcoords], label="CRLite (est.)")
ax.plot([min(p,1-p) for (p,t,b,q) in results_fringe], [b*8/1.e7/ent(p) for (p,t,b,q) in results_fringe], label="Ours (measured, 10M certs)")
ax.plot(xcoords, [new_est(x)/ent(x) for x in xcoords], label="Ours (asymptotic)")
# ax.plot(xcoords, [1 for x in xcoords], label="Shannon")
ax.legend(loc="upper left", prop=prop)
fig.savefig("ratio.png", dpi=600)

print("Max ratio:", max(b*8/1.e7/ent(p) for (p,t,b,q) in results_fringe))
print("Max ratio above 1e-4:", max(b*8/1.e7/ent(p) for (p,t,b,q) in results_fringe if p > 1e-4))

fig,ax = plt.subplots(figsize=(6,4.5))
ax.set_xlim(1e-4,0.5)
ax.set_xlabel("fraction revoked")
ax.set_ylabel(u"bits per revocation")
ax.set_xscale("log")
#ax.set_yscale("log")
ax.plot(xcoords, [crlite(x)/x for x in xcoords], label="CRLite (est.)")
#ax.plot(xcoords, [crlite(x,1) for x in xcoords], label="CRLite/matrix (est.)")
ax.plot([min(p,1-p) for (p,t,b,q) in results_fringe], [b*8/1.e7/p for (p,t,b,q) in results_fringe], label="Ours (measured, 10M certs)")
ax.plot(xcoords, [new_est(x)/x for x in xcoords], label="Ours (asymptotic)")
ax.plot(xcoords, [ent(x)/x for x in xcoords], label="Entropy limit")
ax.legend(loc="upper right", prop=prop)
fig.savefig("entropy.png", dpi=600)

fig,ax = plt.subplots(figsize=(6,4.5))
# ax.set_xlim(10e-5,0.5)
ax.set_xlabel("fraction revoked")
ax.set_ylabel(u"µs / key")
# ax.set_xscale("log")
ax.plot([min(p,1-p) for (p,t,b,q) in results_hier], [t/10 for (p,t,b,q) in results_hier], label="create hier")
ax.plot([min(p,1-p) for (p,t,b,q) in results_fringe], [t/10 for (p,t,b,q) in results_fringe], label="create fringe")
ax.plot([min(p,1-p) for (p,t,b,q) in results_hier], [q/10 for (p,t,b,q) in results_hier], label="query hier")
ax.plot([min(p,1-p) for (p,t,b,q) in results_fringe], [q/10 for (p,t,b,q) in results_fringe], label="query fringe")
ax.legend(loc="upper left", prop=prop)
fig.savefig("sketch-speed.png", dpi=600)
