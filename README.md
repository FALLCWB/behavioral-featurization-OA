# Behavioral Featurization

Companion code for the paper *Behavioral Featurization: Operationalizing Data of
Unknown or Volatile Dimensionality for Machine Learning* (IEEE Access).

Some learning tasks involve an object whose dimensionality is not fixed: the
number or size of its parts is unknown in advance, or changes as the object
evolves, so no fixed-length vector exists to feed a standard model. Behavioral
Featurization derives a fixed family of dimension-agnostic behavioral functionals
from the object's evolution over a window, featurizes them, and learns on the
resulting fixed vector. This repository contains the instrumentation,
data-collection, and analysis code for the three real case studies (system
memory, temporal graphs, ecological communities), the shape-classification and
order-book experiments, the proposition demonstration, and all learned baselines.

## Authors

Filipe A. L. Lemos, Douglas do Amaral, Felipe M. Priotto, Andre E. Lazzaretti.

## Requirements

Python 3.12 (CPU-only is sufficient). Install with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Third-party datasets (SNAP temporal graphs, EcoMon plankton, UCR benchmarks) are
public but not redistributed here; see [DATA.md](DATA.md) for download locations.
The memory case study needs no external dataset, and the order book is collected
from the public Binance API.

## Reproducing the paper

All scripts live in `case-studies/` and are run with the project virtualenv, for
example `python run_stats.py`. The `results_*.json` files record the numbers
behind the reported tables.

| Script | Produces |
| --- | --- |
| `run_stats.py`, `run_rigorous.py` | Source classification across three domains (Table I) |
| `run_contrast_ablation.py` | Comparison with adjacent representations (Table II); spatial-dispersion ablation (Table IV) |
| `run_order_detection.py` | Order-structure detection on real data (Table III) |
| `run_orderbook.py`, `orderbook/` | Limit order book order-detection |
| `run_ucr.py` | Shape classification on UCR benchmarks (Table V) |
| `run_prop2_kernel.py` | Direct demonstration of Proposition (ii): Markov-kernel recovery (Fig. 5) |
| `common/generativity_synthetic.py` | Controlled generativity experiment (Fig. 3) |
| `run_window_sweep.py` | Window-length sweep (Fig. 7) |
| `run_resource.py`, `run_resource_stats.py` | Cost comparison vs. Deep Sets and LSTM (Table VI) |
| `run_resource_extra.py` | Additional learned baselines: attention-MIL, Set Transformer, TCN |
| `run_resource_dyngraph.py` | Learned dynamic-graph baseline (temporal GNN) |
| `run_paired_gaps.py` | Out-of-fold paired-bootstrap 95% CIs for every baseline gap |
| `run_ram_extra.py` | Peak RAM of the added baselines |
| `run_deepsets_gap.py`, `run_ecology_gap.py` | Paired Deep Sets gaps on memory and ecology |
| `run_forecast_sig.py`, `run_forecasting.py` | Forecasting labels and the cluster-bootstrap order effect |
| `run_tsfresh_robustness.py` | Robustness of the conclusions to the featurizer |
| `run_feature_selection.py` | FDR feature selection |
| `make_figures_motivation.py`, `make_cost_figure.py` | Motivation and cost figures |

All learned baselines share one fair protocol (minibatch Adam, gradient clipping,
single-threaded timing) and the same leakage-controlled cross-validation
(non-overlapping windows, stratified group k-fold over contiguous time blocks).

## License

MIT. See [LICENSE](LICENSE).
