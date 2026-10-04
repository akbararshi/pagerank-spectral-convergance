# PageRank Spectral Convergence: Adaptive Tactical Match Intelligence

This repository documents a multi-phase computer science research pipeline, moving from an empirical scaling experiment (**v1-experimental-framework**) into a production-grade, API-driven visual intelligence server (**v2-dynamic-football-engine**). The platform optimizes Google's PageRank algorithm to evaluate structural influence within dynamic multi-agent passing networks, shifting away from rigid static parameters to calculate fluid, absolute volume-weighted network models natively in Python.

---

##  System Architectural Evolution (v1 vs v2)

| Engineering Vector | v1 (The Experimental Framework) | v2 (The Live Automated Engine) |
| :--- | :--- | :--- |
| **Primary Goal** | Benchmarking PageRank limits on graphs from 10 to 5,000 nodes | Live tactical match analysis using real-world football datasets |
| **Data Ingestion** | Hardcoded matrix buffers entered manually into code files | Automated live telemetry tracking via the StatsBomb API |
| **Damping Factor (alpha)** | Manual testing across static bounds (alpha = 0.50 to alpha = 0.99) | Automated Macroscopic Sigmoid Governor mapping match density |
| **Scoring Formula** | Relative percentage-share probability matrices | Absolute weights scaled by player-specific outbound passes |
| **User Interface** | Terminal-driven test script for scalability logging | Full-stack interactive Streamlit web app with autocomplete search |
| **Visual Analytics** | Static numeric text charts | Native Streamlit bar charts paired with a visual Matplotlib network map |

---

## Project Repository Schema

```text
pagerank-spectral-convergence/
├── v1-experimental-framework/
│   ├── data/
│   │   ├── experiment_results.json          # Logged iteration metrics tracking convergence outputs
│   │   ├── simple_graphs.py                 # Synthetic matrix graph generators (Star and Spider-Trap)
│   │   ├── spider_trap_convergence_chart.png # Vector plot charting computation lag in cyclic traps
│   │   └── star_graph_convergence_chart.png  # Vector plot charting scale convergence on hub topologies
│   └── src/
│       ├── engine.py                        # Matrix calculation algorithms testing bounds from α=0.50 to α=0.99
│       ├── generate_plot.py                 # Data visualizer pipeline converting JSON outputs to PNG charts
│       └── main.py                          # Master execution script orchestrating the simulation loop
│
├── v2-dynamic-football-engine/
│   ├── data/
│   │   └── fetcher.py               # StatsBomb API ingestion layer and event log pipeline cleaner
│   └── engine/
│       ├── adaptive_pagerank.py     # Vectorized matrix calculus engine housing the Sigmoid Governor
│       └── app.py                   # Responsive Streamlit analytical app with manual 2026 sandbox mode
│
├── .gitignore                       # Environment tracking file to exclude python system cache artifacts
├── README.md                        # Master portfolio overview and documentation spine
└── requirements.txt                 # Unified repository external library dependency configuration registry
```

---

## 🔬 Phase 1 Deep-Dive: Synthetic Graph Scale Testing

The **v1-experimental-framework** serves as the theoretical mathematical foundation for this portfolio. Before processing real-world sports logs, a simulation environment was constructed to test the convergence behavior of the PageRank algorithm on networks scaled from 10 up to 5,000 nodes using custom topology builders (`data/simple_graphs.py`).

Logging these simulation metrics into `data/experiment_results.json` surfaced a critical behavioral anomaly regarding high-damping factors:

### 1. The Small-Network Signal Flattening Problem
Google’s standard damping factor (alpha = 0.85), while optimal for massive web crawling, introduces severe normalization distortions in smaller systems (like an 11-node team matrix). It flattens out mathematical variance, stripping key playmaker hubs of their true importance.

### 2. The High-Damping Structural Bottleneck
To preserve localized network details, experiments were conducted by manually raising the variable close to its limit (alpha = 0.50 up to alpha = 0.99). While raising α = 0.99 successfully isolated hub significance, it introduced an exponential computational lag during power iterations. 

Empirical testing exposed a **Graph Density Convergence Divergence**: inside a tight 1,000-node cyclic spider trap, the calculation was severely dragged out to **757 iteration loops** to reach stability. However, when scaled to 5,000 nodes, the sheer length of the linear distribution chain thinned out vector differentials, allowing the system to pass tolerance thresholds earlier at **597 iteration loops**.

Mapping these performance boundaries demonstrated that small networks require a dynamic, automated solution rather than a hardcoded fixed variable—directly driving the creation of the **v2 Sigmoid Governor**.

---

## 🧬 Phase 2 Mathematics: Dynamic Matrix Calibration

To resolve the boundary distortions and computational bottlenecks isolated in Phase 1, the **v2-dynamic-football-engine** replaces fixed variables with a dynamic, two-tier network optimization model.

### 1. The Macroscopic Sigmoid Governor
Instead of utilizing an arbitrary, hardcoded damping constant (alpha = 0.85), the v2 engine reads the overall connectivity density of the passing network and automatically adjusts the baseline damping factor (\(\mathcal{D}_{\text{global}}\)) using a custom logistic sigmoid curve:

\[\mathcal{D}_{\text{global}} = 0.85 + \frac{0.13}{1 + \exp\left(-10 \cdot \left(\text{Density} - 0.45\right)\right)}\]

This math ensures the algorithm adapts to the game state: it automatically dampens low-density vertical systems (like direct long-ball teams) to filter out noise, while scaling up to a ceiling of **0.98** for highly dense, technical possession systems to preserve passing signals.

### 2. Player-Specific Absolute Outbound Weighting
Traditional PageRank yields relative percentage shares that add up to 1.0. This creates a severe distortion where a defender in a low-possession team can generate a higher relative rank score than a master playmaker in a high-volume team simply because their local network pool is smaller.

To fix this sample-size distortion, the relative stationary distribution values (\(v_i\)) are scaled into an absolute evaluation matrix weighted by each individual player's outbound completed pass volume:

\[\text{Weighted Score}_i = v_i \cdot (\text{Absolute Outbound Passes}_i)\]

This shifts the preference directly toward high-volume distribution anchors (like center-backs and defensive midfielders recycling possession) over isolated attacking destination nodes.

---

## 🔧 Local Environment Installation & System Execution

To deploy and execute this data analytics pipeline locally, complete the following environment configuration sequence:

### 1. Ingest Core Dependencies
Ensure the terminal path is pointing to the main root project directory and install the pre-configured requirements package:
```bash
pip install -r requirements.txt
```

### 2. Launch the Production Application Server
Navigate into the production workspace directory tree and fire up the full-stack Streamlit engine utilizing local environment routing flags:
```bash
cd v2-dynamic-football-engine
PYTHONPATH=. python3 -m streamlit run engine/app.py
```

---

## 🎯 Production Validation Match Fixtures

Query these real-world historical match profiles using the new predictive autocomplete search fields to verify the algorithm's tactical metrics:

- **Germany vs. Scotland (Euro 2024 - Opening Match):** Validates the cross-network volume weighting matrix. The formula successfully overrides low-volume defensive nodes to crown **Toni Kroos** as the absolute global match MVP, matching an official record of 101 completed passes.
- **Barcelona vs. Juventus (UCL Final 2015):** Validates the Macroscopic Sigmoid Governor. The engine detects the high fluid sequence density of the match to scale the baseline damping coefficient to its mathematical maximum (**0.977**). The framework isolates the true structural engine room of Barcelona's possession triangles, crowning right-back **Dani Alves** as the absolute network MVP.
- **Argentina vs. France (World Cup 2018 - Round of 16):** Validates the structural bottleneck indicators. The pipeline successfully flags direct vertical attackers like **Kylian Mbappé** with elevated Bottleneck Indices (1.84) while mapping Argentina's high-volume, flat lateral backline passing circuits.



