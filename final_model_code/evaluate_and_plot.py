"""
Stage 7: Load all saved model results, generate plots and summary table.
Runs AFTER model_a_pivot.py, model_b_pairwise.py, model_c_gnn.py, bolt_baseline.py
"""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Load pre-saved per-run result lists ───────────────────────────────────────
def load(name):
    with open(f'results/{name}.pkl', 'rb') as f:
        return pickle.load(f)

bolt_xis = load('bolt_xis')
a_xis    = load('model_a_xis')
b_xis    = load('model_b_xis')
c_xis    = load('model_c_xis')
ns       = load('graph_sizes')

COLORS = {'bolt':'#78909C','a':'#1565C0','b':'#2E7D32','c':'#BF360C'}
MLBLS  = {'bolt':'BOLT (baseline)','a':'Model A: Learned Pivot',
          'b':'Model B: Pairwise XGB','c':'Model C: GraphSAGE'}

os.makedirs('outputs', exist_ok=True)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print(f"{'Method':<32} {'Mean ξ':>8} {'Std ξ':>7} {'Min ξ':>8} {'Max ξ':>8}")
print("="*72)
for m, lbl, xis in [('bolt','BOLT (baseline, T=25)',bolt_xis),
                     ('a',   'Model A: Learned Pivot',a_xis),
                     ('b',   'Model B: Pairwise XGBoost',b_xis),
                     ('c',   'Model C: GraphSAGE (GNN)',c_xis)]:
    print(f"{lbl:<32} {np.mean(xis):>8.4f} {np.std(xis):>7.4f} "
          f"{np.min(xis):>8.4f} {np.max(xis):>8.4f}")
print("="*72)
bm = np.mean(bolt_xis)
for m, lbl, xis in [('a','Model A',a_xis),('b','Model B',b_xis),('c','Model C',c_xis)]:
    imp = np.mean(xis)-bm
    print(f"  {lbl}: +{imp*100:.2f}pp over BOLT  (+{imp/bm*100:.1f}%)")

# ── Figure 1: 2×2 summary ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14,10))
fig.patch.set_facecolor('#FAFAFA')
gs  = GridSpec(2,2,figure=fig,hspace=0.40,wspace=0.30)

mk  = ['bolt','a','b','c']
mn  = ['BOLT\n(baseline)','Model A\nLearned Pivot','Model B\nPairwise XGB','Model C\nGraphSAGE']
alx = [bolt_xis,a_xis,b_xis,c_xis]
mns = [np.mean(x) for x in alx]
sts = [np.std(x)  for x in alx]
cls = [COLORS[m]  for m in mk]

# subplot A — mean bar
ax1 = fig.add_subplot(gs[0,0])
bars = ax1.bar(mn, mns, color=cls, width=0.52, zorder=3, edgecolor='white', linewidth=1.4)
ax1.errorbar(mn, mns, yerr=sts, fmt='none', color='#222', capsize=5, linewidth=1.6, zorder=5)
for bar,val,std in zip(bars,mns,sts):
    ax1.text(bar.get_x()+bar.get_width()/2, val+std+0.004, f'{val:.4f}',
             ha='center',va='bottom',fontsize=8.5,fontweight='bold',color='#111')
for i in range(1,4):
    imp=mns[i]-mns[0]
    ax1.annotate(f'+{imp*100:.1f}%',
                 xy=(mn[i], mns[i]+sts[i]+0.017),
                 ha='center',fontsize=8,color='#1B5E20',fontweight='bold')
ax1.axhline(mns[0],color=COLORS['bolt'],linestyle='--',linewidth=1.1,alpha=0.6)
ax1.set_ylim(0.83,0.99); ax1.set_ylabel('Mean Ordering Efficiency ξ',fontsize=10)
ax1.set_title('A. Mean ξ ± 1 std',fontsize=10,fontweight='bold',pad=8)
ax1.grid(axis='y',alpha=0.30,zorder=0); ax1.set_facecolor('#F5F5F5')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# subplot B — scatter vs size
ax2 = fig.add_subplot(gs[0,1])
for m,xis,mk2 in [('bolt',bolt_xis,'o'),('a',a_xis,'s'),('b',b_xis,'^'),('c',c_xis,'D')]:
    ax2.scatter(ns,xis,c=COLORS[m],s=52,label=MLBLS[m],zorder=4,
                alpha=0.82,marker=mk2,edgecolors='white',linewidths=0.5)
    z=np.polyfit(ns,xis,1); p=np.poly1d(z)
    xs=np.linspace(min(ns),max(ns),80)
    ax2.plot(xs,p(xs),color=COLORS[m],linewidth=1.1,alpha=0.40,linestyle='--')
ax2.set_xlabel('Graph size (nodes)',fontsize=10); ax2.set_ylabel('ξ',fontsize=10)
ax2.set_title('B. ξ vs Graph Size',fontsize=10,fontweight='bold',pad=8)
ax2.legend(fontsize=7.5,loc='lower right',framealpha=0.85)
ax2.grid(alpha=0.28,zorder=0); ax2.set_facecolor('#F5F5F5')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# subplot C — per-graph grouped bars
ax3 = fig.add_subplot(gs[1,:])
x=np.arange(len(ns)); w=0.20
ax3.bar(x-1.5*w,bolt_xis,w,color=COLORS['bolt'],label='BOLT',zorder=3,edgecolor='white')
ax3.bar(x-0.5*w,a_xis,   w,color=COLORS['a'],   label='Model A',zorder=3,edgecolor='white')
ax3.bar(x+0.5*w,b_xis,   w,color=COLORS['b'],   label='Model B',zorder=3,edgecolor='white')
ax3.bar(x+1.5*w,c_xis,   w,color=COLORS['c'],   label='Model C (GNN)',zorder=3,edgecolor='white')
ax3.set_xticks(x); ax3.set_xticklabels([f'n={n}' for n in ns],rotation=45,ha='right',fontsize=7.5)
ax3.set_ylabel('ξ',fontsize=10)
ax3.set_title('C. Per-Graph ξ — All Methods vs BOLT Baseline',fontsize=10,fontweight='bold',pad=8)
ax3.legend(fontsize=8.5,loc='lower right',ncol=4,framealpha=0.85)
ax3.set_ylim(0.78,1.03)
ax3.axhline(np.mean(bolt_xis),color=COLORS['bolt'],linestyle='--',linewidth=1.1,alpha=0.55)
ax3.grid(axis='y',alpha=0.28,zorder=0); ax3.set_facecolor('#F5F5F5')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

plt.suptitle('ML-Based Betweenness Ordering — Results vs BOLT (Singh et al. 2017)',
             fontsize=12,fontweight='bold',y=1.01)
plt.savefig('outputs/fig1_main_results.png',dpi=160,bbox_inches='tight',facecolor='#FAFAFA')
print("\nSaved outputs/fig1_main_results.png")

# ── Figure 2: std-dev + improvement ──────────────────────────────────────────
fig2,axes2 = plt.subplots(1,2,figsize=(12,4.5))
fig2.patch.set_facecolor('#FAFAFA')

ax4=axes2[0]
bars4=ax4.bar(mn,sts,color=cls,width=0.52,zorder=3,edgecolor='white',linewidth=1.4)
for bar,val in zip(bars4,sts):
    ax4.text(bar.get_x()+bar.get_width()/2,val+0.0005,f'{val:.4f}',
             ha='center',va='bottom',fontsize=9,fontweight='bold')
ax4.set_ylabel('Std Dev of ξ (lower = more consistent)',fontsize=10)
ax4.set_title('D. Consistency — Std Dev of ξ',fontsize=10,fontweight='bold',pad=8)
ax4.grid(axis='y',alpha=0.30,zorder=0); ax4.set_facecolor('#F5F5F5')
ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

ax5=axes2[1]
imp_a=[a-b for a,b in zip(a_xis,bolt_xis)]
imp_b=[a-b for a,b in zip(b_xis,bolt_xis)]
imp_c=[a-b for a,b in zip(c_xis,bolt_xis)]
x5=np.arange(len(ns)); w5=0.27
ax5.bar(x5-w5, imp_a,w5,color=COLORS['a'],label='Model A',zorder=3,edgecolor='white')
ax5.bar(x5,    imp_b,w5,color=COLORS['b'],label='Model B',zorder=3,edgecolor='white')
ax5.bar(x5+w5, imp_c,w5,color=COLORS['c'],label='Model C',zorder=3,edgecolor='white')
ax5.axhline(0,color='#333',linewidth=0.9)
ax5.set_xticks(x5); ax5.set_xticklabels([f'n={n}' for n in ns],rotation=45,ha='right',fontsize=7)
ax5.set_ylabel('ξ improvement over BOLT',fontsize=10)
ax5.set_title('E. Per-Graph Improvement over BOLT',fontsize=10,fontweight='bold',pad=8)
ax5.legend(fontsize=9,framealpha=0.85)
ax5.grid(axis='y',alpha=0.30,zorder=0); ax5.set_facecolor('#F5F5F5')
ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

plt.tight_layout(pad=2)
plt.savefig('outputs/fig2_analysis.png',dpi=160,bbox_inches='tight',facecolor='#FAFAFA')
print("Saved outputs/fig2_analysis.png")
