# Learning conditional-average dose responses through representation balancing </br><sub><sub>C. Bockel-Rickermann, T. Vanderschueren, J. Berrevoets, T. Verdonck, W. Verbeke [(2023)]([https://arxiv.org/pdf/2309.03731](https://arxiv.org/pdf/2309.03731))</sub></sub>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![ArXiv: 2309.03731](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/pdf/2309.03731)

The is the repo to the manuscript "Using representation balancing to learn conditional-average dose responses from clustered data"

## Repository structure
This repository is organised as follows:
```bash
|- config/  # Config files
|- data/    # Raw data
|- scripts/ # Scripts to execute experiments and create results
|- src/     # Methods, datasets, and metrics
```

## Installing
This repo uses Python 3.10. We have provided a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
Please use the above in a newly created virtual environment to avoid clashing dependencies.

To replicate our experiments, run the scripts in the `scripts/` folder.  
Raw data has to be downloaded from [here](https://www.dropbox.com/scl/fo/9vwoxongw1b2727mkyg6u/AEhZlwyKQeZ572UHymQK82w?rlkey=38bz6fchgsji86ks7a91810gv&st=erqpa4f0&dl=0) and put in the `data/` folder.  

## Citing
Please cite our paper and/or code as follows:

```tex
@article{BockelRickermann2023,
  title={Using representation balancing to learn conditional-average dose responses from clustered data},
  author={Bockel-Rickermann, Christopher and Vanderschueren, Toon and Berrevoets, Jeroen and Verdonck, Tim and Verbeke, Wouter},
  journal={arXiv preprint arXiv:2309.03731},
  year={2023}
}
```
