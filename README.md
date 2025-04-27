# Flexible and Fast Diagonal Schemes for Simple Bilevel Optimization. Example code

This repository contains the experimental source code to reproduce the numerical experiments in:

* R. I. Bot, E. Chenchene, R. Csetnek, D. Hulett. Flexible and Fast Diagonal Schemes for Simple Bilevel Optimization. 2025. [ArXiv preprint](https://arxiv.org/abs/XXXX.YYYY)

To reproduce the results of the numerical experiments run:
```bash
python3 main.py
```
**Note:** To run `experiment_logistic` a `sklearn` is required.

If you find this code useful, please cite the above-mentioned paper:
```BibTeX
@article{acgn25,
  author = {Bo\c{t}, Radu and Chenchene, Enis and Csetnek, Robert and Hulett, David},
  title = {Flexible and Fast Diagonal Schemes for Simple Bilevel Optimization},
  pages = {XXXX.YYYYY},
  journal = {ArXiv},
  year = {2025}
}
```

## Requirements

Please make sure to have the following Python modules installed:

* [numpy>=1.26.4](https://pypi.org/project/numpy/)
* [numpy>=1.13.1](https://pypi.org/project/scipy/)
* [matplotlib>=3.3.4](https://pypi.org/project/matplotlib/)
* [scikit-learn>=1.4.2](https://scikit-learn.org)


## License
This project is licensed under the GPLv3 license - see [LICENSE](LICENSE) for details.
