# Betweenness Ordering — ML Improvement Project
## Extending BOLT (Singh et al. 2017) with Machine Learning

---

## Folder Structure

```
bc_project/
├── README.md                  ← You are here
├── requirements.txt           ← Python packages to install
├── run_all.py                 ← Master runner (runs everything in order)
│
├── generate_data.py           ← Stage 1+2 : graph generation + exact BC
├── features.py                ← Stage 3   : feature extraction (shared)
├── bolt_baseline.py           ← Stage 5   : BOLT baseline evaluation
├── model_a_pivot.py           ← Stage 4a  : Model A — Learned Pivot
├── model_b_pairwise.py        ← Stage 4b  : Model B — Pairwise XGBoost
├── model_c_gnn.py             ← Stage 4c  : Model C — GraphSAGE GNN
├── evaluate_and_plot.py       ← Stage 6+7 : plots + summary table
│
├── data/                      ← Created automatically
│   ├── train_data.pkl         ← 96 training graphs + exact BC
│   └── test_data.pkl          ← 24 test graphs + exact BC
│
├── models/                    ← Created automatically
│   ├── model_a.pkl            ← Trained XGBoost pivot scorer
│   ├── model_b.pkl            ← Trained XGBoost pairwise ranker
│   └── model_c.pkl            ← Trained GraphSAGE weights
│
├── results/                   ← Created automatically
│   ├── bolt_xis.pkl           ← BOLT ordering efficiency per test graph
│   ├── model_a_xis.pkl        ← Model A efficiency per test graph
│   ├── model_b_xis.pkl        ← Model B efficiency per test graph
│   ├── model_c_xis.pkl        ← Model C efficiency per test graph
│   └── graph_sizes.pkl        ← Node counts for plotting
│
└── outputs/                   ← Created automatically
    ├── fig1_main_results.png  ← Main comparison figure
    └── fig2_analysis.png      ← Std dev + per-graph improvement
```

---

## Setup (One-Time)

### Step 1 — Install Python (if not already)
Download Python 3.10 or 3.11 from https://python.org
During install: CHECK "Add Python to PATH"

### Step 2 — Open terminal / command prompt
- Windows: search "cmd" or "PowerShell"
- Mac/Linux: open Terminal

### Step 3 — Navigate to project folder
```
cd path/to/bc_project
```

### Step 4 — Install required packages
```
pip install -r requirements.txt
```
This installs: networkx, numpy, scikit-learn, xgboost, matplotlib, scipy
Takes ~2 minutes.

---

## Running the Project

### Option A — Run everything at once (recommended)
```
python run_all.py
```
Total time: ~15-20 minutes on a laptop.

### Option B — Run each stage separately (to see each step)
```
python generate_data.py       # ~10-15 min  — creates data/ folder
python bolt_baseline.py       # ~5 min      — BOLT baseline numbers
python model_a_pivot.py       # ~8 min      — trains + tests Model A
python model_b_pairwise.py    # ~3 min      — trains + tests Model B
python model_c_gnn.py         # ~5 min      — trains + tests Model C
python evaluate_and_plot.py   #  <1 min     — plots + summary table
```

### What you will see
Each script prints results as it runs, e.g.:
```
  ER n=150 param=5 -> ξ=0.9350
  ER n=250 param=3 -> ξ=0.9120
  ...
  Model A mean ξ = 0.9433  std = 0.0210
```

At the end, evaluate_and_plot.py prints the final comparison table:
```
Method                   Mean ξ   Std ξ   Min ξ   Max ξ
BOLT (baseline, T=25)    0.8883  0.0352  0.7720  0.9360
Model A: Learned Pivot   0.9433  0.0210  0.8880  0.9760
Model B: Pairwise XGB    0.9180  0.0279  0.8360  0.9560
Model C: GraphSAGE       0.9303  0.0330  0.8520  0.9760
```
And saves two PNG figures to outputs/

---

## Expected Results (approximate — varies slightly each run)

| Method              | Mean ξ | Improvement over BOLT |
|---------------------|--------|----------------------|
| BOLT (baseline)     | ~0.89  | —                    |
| Model A             | ~0.94  | +5-6%                |
| Model B             | ~0.92  | +3-4%                |
| Model C (GraphSAGE) | ~0.93  | +4-5%                |

---

## Troubleshooting

**"Module not found" error:**
Run: `pip install networkx numpy scikit-learn xgboost matplotlib`

**Script runs very slowly (>30 min):**
The graphs in generate_data.py have sizes up to n=400.
To make it faster, open generate_data.py and change:
  `sizes = [80, 150, 250, 400]`  →  `sizes = [80, 150, 200]`

**"File not found: data/train_data.pkl":**
You must run generate_data.py before the model scripts.


The Big Picture — What Model A Does
BOLT uses this hand-crafted formula to pick pivots:
p_i ∝ λ^(-d(v,i)) / deg(i)
Model A throws away this formula and replaces it with a trained XGBoost model that learns which pivots are best from data. Everything else (the sampling, the estimation) stays exactly the same as BOLT.

Stage 1 — Building the Training Dataset
Dataset: 108,936 samples, 13 features
This is the most important part to understand. Here's exactly how those 108,936 samples were created.
For each of the 96 training graphs, we picked 6 random target nodes (nodes whose BC we want to estimate). For each target node v, we looked at every other node i in the graph as a candidate pivot.
For a graph with 150 nodes, that gives:
6 target nodes × 149 candidate pivots = 894 samples from one graph
96 graphs × ~894 = ~85,000+ samples
(larger graphs contribute more, totalling 108,936)
What is one sample?
Each sample is one (target node v, candidate pivot i) pair. It has:
Input X — 13 features describing the relationship between v and i:
#FeatureWhat it captures0d(v,i) / nNormalized BFS distance from target to pivot1λ^(-d)The EDDBM distance term — baseline signal21/deg(i)Inverse degree of pivot31/deg(v)Inverse degree of target4deg(i)/avg_degHow "hub-like" the pivot is5clustering(i)Local triangle density around pivot6kcore(i)How deep in dense core the pivot sits71/siblingsInverse count of nodes at same BFS level8clustering(v)Clustering of target node9kcore(v)K-core of target node10d==1 flagIs pivot a direct neighbor?11d==2 flagIs pivot 2 hops away?12log(d)Smooth distance signal
Features 0, 1, 2, 3 — BOLT already uses these implicitly in its formula. Features 5, 6, 7, 8, 9 — clustering and k-core — are completely new. These are what BOLT is blind to.
Output Y — the ground truth pivot quality:
Y = log(1 + δ_{i•}(v))
Where δ_{i•}(v) is the dependency of pivot i on target v — how much of pivot i's shortest-path traffic passes through v. This was computed exactly using Brandes during generate_data.py.
This is the optimal pivot score — from Theorem 1 of the paper, if you sample pivots proportional to their actual δ_{i•}(v), you get exact BC in one sample. So we're teaching the model to approximate this optimal distribution.

Stage 2 — Training XGBoost
Train R² = 0.8622
XGBoost builds an ensemble of 250 decision trees, each one correcting the errors of the previous one (this is called gradient boosting).
The training process:
Sample 1: features=[0.02, 0.94, 0.12, ...] → predicted=0.34, actual=0.41, error=-0.07
Sample 2: features=[0.15, 0.61, 0.08, ...] → predicted=1.20, actual=1.18, error=+0.02
...
Tree 1 learns rough pattern
Tree 2 corrects Tree 1's mistakes
Tree 3 corrects remaining mistakes
...
Tree 250 → final model
R² = 0.8622 means the model explains 86.22% of the variance in pivot quality scores. Not perfect, but good enough to pick much better pivots than EDDBM's fixed formula.
The trained model gets saved:
models/model_a.pkl   ← 250 decision trees, ~few MB

Stage 3 — Inference (Test Time)
This is what happens for each of the 24 test graphs when we evaluate:
For pair (u, v) — which has higher BC?

Step 1: Run one BFS from u
        → get distance d(u, i) for every node i
        → extract 13 features for every (u, i) pair
        → model_a.predict() on all of them at once
        → get scores s_i for every candidate pivot i
        → normalize: p_i = s_i / sum(s_j)   ← learned probabilities

Step 2: Sample T=25 pivots using learned probabilities p_i
        For each pivot:
            run BFS from pivot
            compute δ_{pivot→}(u)   ← how much traffic through u
            add δ/p_i to running sum

Step 3: BC_estimate(u) = running_sum / 25

Step 4: Repeat Steps 1-3 for v → BC_estimate(v)

Step 5: Return BC_estimate(u) > BC_estimate(v)
The key difference from BOLT — in Step 1, instead of:
python# BOLT does this:
p_i = lambda^(-distance) / degree(i)

# Model A does this:
p_i = xgboost_model.predict(13_features_for_i)

Input/Output Summary
INPUT  (training):  96 graphs + exact BC values
                    → 108,936 (pivot_features, δ_score) pairs

TRAINING:           XGBoost fits 250 trees on these pairs
                    → learns: "which structural features predict
                               a good pivot?"

OUTPUT (model):     models/model_a.pkl
                    → a function: 13 features → pivot quality score

INPUT  (testing):   24 unseen test graphs
                    → for each graph, extract features on the fly

OUTPUT (result):    ξ per graph → mean ξ = 0.9433

Why Model A beats BOLT
Look at this comparison on the hardest graph (n=250, dense):
BOLT     n=250 param=10 → ξ=0.7720   ← really struggled
Model A  n=250 param=10 → ξ=0.8880   ← much better








Stage 1 — Building the Training Dataset
Dataset: 57,600 pairs, 16 features
Class balance: 0.500
From the 96 training graphs, we sampled 600 random pairs (u,v) per graph where BC(u) ≠ BC(v).
96 graphs × 600 pairs = 57,600 training samples
What is one sample?
For a pair (u, v), we extract the 8 node features for u and the 8 node features for v separately, then combine them:
feat(u) = [deg_norm, clustering, kcore, avg_nbr_deg,
           log_deg, is_leaf, deg_ratio, eccentricity]
           
feat(v) = [same 8 features for v]

Input X = [feat(u) - feat(v)]          ← 8 numbers (difference)
        ∥ [|feat(u) - feat(v)|]        ← 8 numbers (absolute difference)
        = 16 numbers total
Why concatenate difference AND absolute difference?
feat(u) - feat(v)    tells the model DIRECTION
                     e.g. u has higher degree than v → positive number
                     
|feat(u) - feat(v)|  tells the model MAGNITUDE
                     e.g. how different are they, regardless of who is bigger
Output Y — binary label:
Y = 1   if BC(u) > BC(v)   ← u is more central
Y = 0   if BC(u) < BC(v)   ← v is more central
Class balance = 0.500 means exactly half the pairs have u winning, half have v winning. This is by construction — we randomly assign which node is u and which is v.

Stage 2 — Training XGBoost Classifier
Validation accuracy: 0.9047
XGBoost trains 300 decision trees. But this time it's classification not regression — the model learns a decision boundary in 16-dimensional space:
"If degree difference is large AND clustering difference points this way
 AND kcore difference is large → then the higher-degree, lower-clustering,
 higher-kcore node almost certainly has higher BC"
15% of data (8,640 pairs) was held out for validation. The model never saw these during training. It got 90.47% correct on them — that's the honest accuracy before we even test on new graphs.
The model is saved:
models/model_b.pkl   ← 300 decision trees

Stage 3 — Inference (Test Time)
This is where Model B is fundamentally faster than BOLT and Model A:
BOLT / Model A inference for one pair:
    BFS from u (O(m)) × 25 times = 25 BFS traversals
    BFS from v (O(m)) × 25 times = 25 BFS traversals
    Total: 50 BFS traversals per pair
    
    
    
    
Model B inference for one pair:
    Extract 8 features for u — one BFS (O(m))
    Extract 8 features for v — one BFS (O(m))
    Compute 16-dim difference vector — O(1)
    model_b.predict(16 numbers) — O(1)
    Total: 2 BFS traversals per pair
Model B needs 25× fewer BFS traversals than BOLT per pair.

Why 57,600 pairs vs 108,936 samples in Model A?
Model A: 6 target nodes × all_other_nodes × 96 graphs = 108,936
         (needs many pivot candidates per target)

Model B: 600 pairs × 96 graphs = 57,600
         (just needs pairs of nodes, much simpler)
Model B's training data is simpler to construct — you just need pairs of nodes and their BC ordering. No need to compute individual pivot dependencies.

The 16 Features — Why This Works
The key insight is that the features that make a node have high BC are well understood:
High BC node typically has:
✓ High degree           (many connections)
✓ Low clustering        (neighbors don't connect to each other — it's a bridge)
✓ High k-core number   (sits deep in the dense core)
✓ High avg_neighbor_degree
✓ Low eccentricity     (centrally located)
So if node u has much higher degree, much lower clustering, and much higher k-core than v, the model confidently says BC(u) > BC(v).
The difference vector directly encodes these comparisons:
feat(u) - feat(v) = [+0.4, -0.3, +0.5, +0.2, +0.3, 0.0, +0.6, -0.1]
                      ↑      ↑     ↑
                   u has  u less  u deeper   → u likely has higher BC
                   higher clustered in core
                   degree

Full Input/Output Summary
INPUT  (training):
    96 graphs + exact BC
    → 57,600 pairs (u,v) each with:
       X = 16-dim difference feature vector
       Y = 1 if BC(u)>BC(v), else 0

TRAINING:
    XGBoost classifier, 300 trees
    Learns decision boundary in 16-dim space
    Validation accuracy: 90.47%
    → saves models/model_b.pkl

INPUT  (testing):
    24 unseen graphs
    250 pairs per graph

INFERENCE per pair (u,v):
    extract feat(u), feat(v)  [structural features]
    diff = feat(u) - feat(v)
    absdiff = |feat(u) - feat(v)|
    x = concat(diff, absdiff)   [16 numbers]
    prediction = model_b.predict(x)   [0 or 1]

OUTPUT:
    ξ per graph → mean ξ = 0.9180





The Core Idea — Why a Neural Network on Graphs?
Model A and Model B both use hand-extracted features — we manually decided what to measure (degree, clustering, k-core etc.). A GNN learns its own representation of each node by looking at the actual graph structure — its neighbors, neighbors-of-neighbors, and so on.
Model A/B thinking:
"I measured degree=5, clustering=0.3, kcore=2 for this node"
→ human decided these features matter

Model C thinking:
"This node connects to these specific neighbors, who connect to 
 these specific nodes, who connect to..."
→ model learns what matters from the graph structure itself

What is GraphSAGE?
GraphSAGE = Graph SAmple and agGrEgate. The key idea is message passing — every node collects information from its neighbors and updates its own representation.
Think of it like gossip spreading through a network:
Round 1 (Layer 1):
  Every node tells its neighbors: "here is my basic info"
  Every node listens to all its neighbors and updates itself
  → each node now knows about its 1-hop neighborhood

Round 2 (Layer 2):
  Same process again with updated representations
  → each node now knows about its 2-hop neighborhood

Round 3 (Layer 3):
  Same process again
  → each node now knows about its 3-hop neighborhood
After 3 rounds, each node's embedding (a vector of 16 numbers) captures the structure of its entire 3-hop neighborhood — which is exactly what determines betweenness centrality.

The Architecture — Layer by Layer
Input:  8 features per node
        [degree, clustering, kcore, avg_nbr_deg,
         log_deg, is_leaf, deg_ratio, eccentricity]

Layer 1 (SAGELayer):  8 → 32 dimensions
Layer 2 (SAGELayer): 32 → 32 dimensions  
Layer 3 (SAGELayer): 32 → 16 dimensions

Output: 16-dimensional embedding per node
Each SAGELayer does this mathematical operation:
h_v = ReLU( W_self × h_v  +  W_neigh × mean(h_neighbors) + bias )
       ↑           ↑                    ↑
    activate   my own info    average of all neighbors' info
Where:

W_self and W_neigh are weight matrices the model learns during training
ReLU sets negative values to zero (adds non-linearity)
After each layer, embeddings are L2-normalized (divided by their length) to prevent values exploding


Stage 1 — Training Data
Training: 96 graphs, input_dim=8, epochs=40
From each training graph, we sampled 120 random pairs (u, v):
96 graphs × 120 pairs = ~11,520 training pairs
For each pair the label is:
Y = 1.0  if BC(u) > BC(v)
Y = 0.0  if BC(u) < BC(v)

Stage 2 — Training Loop (40 Epochs)
Epoch   1/40  loss=0.4712  train_acc=0.8534
Epoch  10/40  loss=0.2033  train_acc=0.9195
Epoch  20/40  loss=0.1885  train_acc=0.9236
Epoch  30/40  loss=0.1799  train_acc=0.9273
Epoch  40/40  loss=0.1717  train_acc=0.9310
Each epoch means the model has seen all 96 training graphs once. Here is exactly what happens in one epoch:
Forward Pass (computing the prediction)
Step 1: Start with 8 raw features per node
        node_0: [0.33, 0.12, 0.50, 0.44, 0.41, 0, 1.2, 0.03]
        node_1: [0.67, 0.05, 0.80, 0.71, 0.68, 0, 2.1, 0.02]
        ...

Step 2: Layer 1 message passing
        For each node v:
            collect embeddings of all neighbors
            take their mean → "neighborhood summary"
            combine: h_v = ReLU(W_self×h_v + W_neigh×mean_neighbors + b)
        Now each node has 32 numbers instead of 8

Step 3: Layer 2 message passing (same process, 32→32)
        Now each node's 32 numbers encode 2-hop neighborhood

Step 4: Layer 3 message passing (32→16)
        Now each node has 16-dimensional embedding
        encoding its entire 3-hop neighborhood structure

Step 5: For pair (u, v):
        diff    = embed(u) - embed(v)          [16 numbers]
        absdiff = |embed(u) - embed(v)|        [16 numbers]
        combined = concat(diff, absdiff)        [32 numbers]
        logit    = W_head × combined + bias     [1 number]
        prob     = sigmoid(logit)               [0 to 1]
        
        prob > 0.5 → predict BC(u) > BC(v)
Loss Computation
loss = -(Y × log(prob) + (1-Y) × log(1-prob))
This is binary cross-entropy. Intuitively:

If Y=1 (u has higher BC) and prob=0.9 → small loss (good prediction)
If Y=1 (u has higher BC) and prob=0.2 → large loss (bad prediction)

Backward Pass (learning from mistakes)
The loss gets propagated backwards through all layers to compute gradients — how much should each weight change to reduce the loss? This is backpropagation, implemented manually in our code with NumPy.
loss → gradient flows back through ranking head
     → gradient flows back through Layer 3
     → gradient flows back through Layer 2 
     → gradient flows back through Layer 1
     → all weights updated by Adam optimizer
Adam Optimizer Update
For each weight W, Adam keeps track of:
m = momentum    (running average of gradients)
v = velocity    (running average of squared gradients)

W = W - lr × (m / sqrt(v))
This is smarter than plain gradient descent — it adapts the learning rate for each weight individually.

What the Numbers Tell You
Epoch  1: loss=0.4712  train_acc=0.8534
First pass through all graphs. Weights are random (Glorot initialization). Already 85% accurate because even random weights with the right architecture learn something from graph structure.
Epoch 10: loss=0.2033  train_acc=0.9195
Loss dropped from 0.47 to 0.20 — model learned fast. Accuracy jumped to 91.95%.
Epoch 40: loss=0.1717  train_acc=0.9310
Converging — loss barely changing between epoch 30 and 40. Training accuracy 93.1%. Model is saved here.

Stage 3 — Inference (Test Time)
This is where GNN is dramatically faster than everything else:
BOLT/Model A per pair:
    50 BFS traversals (25 per node)

Model B per pair:
    2 BFS traversals (features for u and v)

Model C per pair:
    Embeddings computed ONCE for entire graph
    Then per pair: just one dot product (O(1))
Here is exactly what happens:
Test graph arrives (n=150 nodes)

Step 1: Compute embeddings for ALL 150 nodes at once
        → 3 rounds of message passing over the whole graph
        → result: 150 × 16 matrix (one 16-dim vector per node)
        Time: one pass over all edges = O(m)

Step 2: For each of 250 test pairs (u, v):
        embed_u = embeddings[u]    ← just lookup, O(1)
        embed_v = embeddings[v]    ← just lookup, O(1)
        prob = sigmoid(W_head × concat(embed_u-embed_v, |embed_u-embed_v|))
        predict = prob > 0.5
        Time: O(1) per pair

Total: O(m) once + O(1) × 250 pairs
This is why you see times like 0.01s per graph for Model C — compared to minutes for BOLT.

Why "from scratch" matters
We implemented GraphSAGE using only NumPy — no PyTorch, no TensorFlow. This means we manually coded:
✓ Forward pass through 3 SAGE layers
✓ Mean aggregation over neighbors
✓ L2 normalization
✓ Sigmoid activation
✓ Binary cross-entropy loss
✓ Backpropagation through all layers
✓ Gradient routing through mean aggregation
✓ Adam optimizer with momentum and velocity
✓ Glorot weight initialization
This is significant to show your professor — you're not just calling model.fit(), you understand the mathematics deeply enough to implement it manually.

Full Input/Output Summary
INPUT (training):
    96 graphs, each with:
    - adjacency structure (who connects to whom)
    - 8 node features per node
    - exact BC values (ground truth)
    → ~11,520 (u,v,label) training pairs

TRAINING:
    3-layer GraphSAGE with message passing
    40 epochs, Adam optimizer
    Binary cross-entropy loss
    Manual backpropagation in NumPy
    → saves models/model_c.pkl

INPUT (testing):
    24 unseen graphs
    250 pairs per graph

INFERENCE per graph:
    One forward pass → embeddings for ALL nodes
    Per pair: lookup + dot product → prediction

OUTPUT:
    ξ per graph → mean ξ = 0.9318

All Four Models Compared
MethodMean ξStdKey ideaInference costBOLT0.88830.0352Hand-crafted EDDBM formula50 BFS per pairModel A0.94330.0210Learned pivot scores50 BFS per pairModel B0.91800.0279Pairwise classification2 BFS per pairModel C0.93180.0329GNN message passingO(m) once then O(1)
Model A wins on accuracy. Model C wins on inference speed after the
