"""Five industrial members, one procedure. The boundary moves; the shape does not.

Two fabs and three inventory members, the last of which is the thirty-eight
supply chains Willems published. The generated inventory members check that the
procedure carries outside scheduling; the real one is the result about
inventory.
"""
import json, os
import matplotlib; import numpy as np
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif",
 "font.sans-serif":["Helvetica","Arial","DejaVu Sans"],
 "font.size":8,"axes.titlesize":9,"axes.labelsize":8,
 "xtick.labelsize":7.5,"ytick.labelsize":7.5,
 "axes.spines.top":False,"axes.spines.right":False,
 "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
 "grid.color":"#d9d9d9","grid.linewidth":0.6,"legend.frameon":False,
 "figure.dpi":200,"savefig.dpi":400,"savefig.bbox":"tight"})
MUTED,INK="#8a8985","#0b0b0b"; TH=0.5
H=os.path.dirname(os.path.abspath(__file__))
R=os.path.join(H,"..","RESILIENCE-SUPER-P")

def from_jsonl(f):
    rows=[json.loads(l) for l in open(os.path.join(H,f))][1:]
    return rows
def from_json(f):
    return json.load(open(os.path.join(R,f)))
def grid(rows):
    Ws=sorted({r["width"] for r in rows}); Ts=sorted({r["T"] for r in rows})
    S=np.array([[np.mean([r["sigma"] for r in rows
                if r["width"]==w and r["T"]==T]) for T in Ts] for w in Ws])
    P=np.array([[r["P"] for r in rows if r["width"]==w][0] for w in Ws])
    return P,np.array(Ts),S
def cross(x,y):
    for i in range(1,len(x)):
        if y[i-1]<TH<=y[i]:
            f=(TH-y[i-1])/(y[i]-y[i-1])
            return 10**(np.log10(x[i-1])+f*(np.log10(x[i])-np.log10(x[i-1])))
    return None

BARS=[]
M=[("fab, ten products", from_jsonl("imitate.jsonl"), "#2a78d6", "-"),
   ("fab, two products",  from_jsonl("imitate_hvlm.jsonl"), "#1b4f8a", "-"),
   ("safety stock, 3 layers", from_json("sweep_gsm.json"), "#eb6834","--"),
   ("safety stock, 6 layers", from_json("sweep_gsm_L6.json"), "#a8391a","--"),
   ("thirty-eight real chains", from_json("sweep_real.json"), "#1baf7a","--")]

fig,ax=plt.subplots(1,3,figsize=(7.8,2.6),
                    gridspec_kw=dict(width_ratios=[1,1,1.1]))
for name,rows,c,ls in M:
    P,Ts,S=grid(rows)
    for i,p in enumerate(P):
        lw,al=(2.0,1.0) if S[i].max()>=TH else (1.0,0.3)
        ax[0].plot(Ts/Ts.max(),S[i],color=c,ls=ls,lw=lw,alpha=al,
                   marker="o",ms=2.8,mec="white",mew=0.5)
    ax[1].plot(P,S[:,-1],color=c,ls=ls,lw=2.0,marker="o",ms=4,
               mec="white",mew=0.7,label=name)
    pc=cross(P,S[:,-1])
    if pc: ax[1].plot([pc],[TH],marker="v",ms=6,color=c,mec="white",mew=0.7)
    BARS.append((name, pc, c))
for a,t,xl in ((ax[0],"a   the budget axis","fraction of the budget swept"),
               (ax[1],"b   the program axis","parameters")):
    a.axhline(TH,color=MUTED,lw=0.9,ls=(0,(3,2)))
    a.set_xscale("log"); a.set_xlabel(xl); a.set_ylim(-0.03,1.03)
    a.set_title(t,loc="left",pad=4); a.grid(True); a.set_axisbelow(True)
ax[0].set_ylabel("instances beating the incumbent")
ax[0].annotate("bold: scales that cross",xy=(0.012,0.94),fontsize=6.6,color=INK)
ax[1].legend(fontsize=6.2,loc="upper left",handlelength=1.5)
ax[1].annotate("$\\blacktriangledown$ the boundary",xy=(300,0.55),
               fontsize=6.6,color=INK)
y=np.arange(len(BARS))
ax[2].barh(y,[b[1] or 0 for b in BARS],color=[b[2] for b in BARS],height=0.6)
for k,(nm,pc,c) in enumerate(BARS):
    ax[2].annotate(f"{pc:,.0f}" if pc else "none",xy=(pc or 1,k),
                   xytext=(4,0),textcoords="offset points",va="center",
                   fontsize=7,color=c,fontweight="bold")
ax[2].set_yticks(y); ax[2].set_yticklabels(
    ["fab, ten products","fab, two products","stock, 3 layers",
     "stock, 6 layers","38 real chains"],fontsize=6.8)
ax[2].invert_yaxis(); ax[2].set_xscale("log"); ax[2].set_xlim(10,1e5)
ax[2].set_xlabel("parameters at the boundary")
ax[2].set_title("c   where each crosses",loc="left",pad=4)
ax[2].grid(axis="x"); ax[2].set_axisbelow(True)
fig.tight_layout(w_pad=1.7)
o=os.path.join(H,"figures","members")
fig.savefig(o+".pdf"); fig.savefig(o+".png"); plt.close(fig)
for name,rows,_,_ in M:
    P,Ts,S=grid(rows); print(f"{name:>24}: P_crit = "
        + (f"{cross(P,S[:,-1]):,.0f}" if cross(P,S[:,-1]) else "none"),flush=True)
print("wrote figures/members.pdf")
