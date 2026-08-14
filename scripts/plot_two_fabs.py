"""Both fab types on one figure. The boundary moves; its shape does not."""
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
INK,MUTED = "#0b0b0b","#8a8985"
HERE=os.path.dirname(os.path.abspath(__file__))
TH=0.5

def load(f):
    rows=[json.loads(l) for l in open(os.path.join(HERE,f))][1:]
    Ws=sorted({r["width"] for r in rows}); Ts=sorted({r["T"] for r in rows})
    S=np.array([[np.mean([r["sigma"] for r in rows
                 if r["width"]==w and r["T"]==T]) for T in Ts] for w in Ws])
    P=np.array([[r["P"] for r in rows if r["width"]==w][0] for w in Ws])
    return P,np.array(Ts),S

def cross(x,y,th=TH):
    for i in range(1,len(x)):
        if y[i-1]<th<=y[i]:
            f=(th-y[i-1])/(y[i]-y[i-1])
            return 10**(np.log10(x[i-1])+f*(np.log10(x[i])-np.log10(x[i-1])))
    return None

fabs=[("ten products, low volume","imitate.jsonl","#2a78d6"),
      ("two products, high volume","imitate_hvlm.jsonl","#eb6834")]
fig,ax=plt.subplots(1,3,figsize=(7.6,2.6),
                    gridspec_kw=dict(width_ratios=[1,1,1.05]))
for name,f,c in fabs:
    P,Ts,S=load(f)
    a=ax[0]
    for i,p in enumerate(P):
        lw,al=(2.2,1.0) if S[i].max()>=TH else (1.1,0.35)
        a.plot(Ts,S[i],color=c,lw=lw,alpha=al,marker="o",ms=3.2,
               mec="white",mew=0.6,label=name if i==len(P)-1 else None)
    b=ax[1]
    j=len(Ts)-1
    b.plot(P,S[:,j],color=c,lw=2.2,marker="o",ms=4,mec="white",mew=0.7,
           label=name)
    pc=cross(P,S[:,j])
    if pc:
        b.axvline(pc,color=c,lw=0.9,ls=(0,(3,2)))
        b.annotate(f"{pc:,.0f}",xy=(pc,0.06),fontsize=6.6,color=c,
                   ha="center",rotation=90)
    ax[2].plot(P,S[:,j],color=c,lw=2.2,marker="o",ms=4,mec="white",mew=0.7)
for a_,t,xl in ((ax[0],"a   the budget axis","labelled decisions seen"),
                (ax[1],"b   the program axis","parameters"),
                (ax[2],"c   both floors","parameters")):
    a_.axhline(TH,color=MUTED,lw=0.9,ls=(0,(3,2)))
    a_.set_xscale("log"); a_.set_xlabel(xl); a_.set_ylim(-0.03,1.03)
    a_.set_title(t,loc="left",pad=4); a_.grid(True); a_.set_axisbelow(True)
ax[0].set_ylabel("instances where it beats the rules")
ax[0].annotate("bold: the scales that cross",xy=(Ts[0],0.93),fontsize=6.6,
               color=INK)
ax[1].legend(fontsize=6.6,loc="upper left",handlelength=1.4)
ax[2].set_ylim(-0.01,0.2)
ax[2].annotate("neither fab succeeds\nat any size until it does",
               xy=(P[0],0.15),fontsize=6.6,color=MUTED)
fig.tight_layout(w_pad=1.8)
out=os.path.join(HERE,"figures","two_fabs")
fig.savefig(out+".pdf"); fig.savefig(out+".png"); plt.close(fig)
for name,f,_ in fabs:
    P,Ts,S=load(f)
    pc=cross(P,S[:,-1]); tc=None
    for i in range(len(P)):
        t=cross(Ts,S[i])
        if t and tc is None: tc=(P[i],t)
    print(f"{name}: P_crit = {pc:,.0f}" if pc else f"{name}: no crossing",
          f"| first T crossing at {tc[0]:,} params, T={tc[1]:,.0f}" if tc else "",
          flush=True)
print("wrote figures/two_fabs.pdf")
