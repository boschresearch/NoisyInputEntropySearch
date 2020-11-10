# Noisy-Input Entropy Search

This is the companion code for the paper *Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization* by Lukas P. Fröhlich et al., AISTATS 2020. The paper can be found [here](http://proceedings.mlr.press/v108/frohlich20a.html). The code allows users to experiment with the provided acquisition function. Please cite the above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication cited above. It will neither be maintained nor monitored in any way.

## Installation guide

In the root directory of the repository execture the following commands:

```shell
conda env create --file=environment.yaml
conda activate nes
pip install -e .
```

## Example

To run a comparison of different acquisition functions on the synthetic benchmark functions, execute the following:

```shell
python run_experiments.py
```

This automatically creates a sub-directory in the `Results/` directory. To visualize the results, adapt the path in `plot_results.py` and execute it via

```shell
python plot_results.py
```

## License

Noisy-Input Entropy Search is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE.md) file for details.

