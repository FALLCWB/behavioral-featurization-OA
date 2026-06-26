# Datasets

The third-party datasets are public but are not redistributed here. Each carries
its own license and citation. Download them into the per-domain `data/` folders
shown below.

## Temporal graphs (Stanford Large Network Dataset Collection, SNAP)
Files: `CollegeMsg.txt`, `email-Eu-core-temporal.txt`, `sx-mathoverflow.txt`.
Source: https://snap.stanford.edu/data/ (Paranjape, Benson, Leskovec, WSDM 2017).
Place in `case-studies/graphs-dynamic/data/`.

## Ecological communities (EcoMon plankton)
File: `ecomon.csv` (EcoMon plankton counts, 1977-2015).
Source: BCO-DMO dataset 3327, DOI 10.26008/1912/bco-dmo.3327.3.1.
Place in `case-studies/ecology-ecomon/data/`.

## Shape-classification benchmarks (UCR Time Series Archive)
Datasets: ECG200, GunPoint, ItalyPowerDemand (official train/test split, .arff/.ts).
Source: https://www.timeseriesclassification.com / UCR Archive (Dau et al., 2019).
Place in `case-studies/ucr/` (or the path expected by `run_ucr.py`).

## Limit order book
Collected from the public Binance market-data API with
`case-studies/orderbook/collect_orderbook.py` (no API key required).

## Memory case study
No external dataset: `run_*` for the memory domain instruments a live process
through `/proc/self/mem` over an `mmap` arena.
