# Generic Data-driven Reservoir Operation Model (GDROM)

## 1. Overview

This repository contains scripts for running our generic data-driven reservoir operation model (GDROM), which is published on Advances in Water Resources. The GDROM couples hidden Markov Model and decision tree to extract representative operation modules and model the module applications. Please refer to [our paper](https://doi.org/10.1016/j.advwatres.2022.104274) for model details.

We have applied the GDROM to 452 large reservoirs with long-term operation records available across the Contiguous United States (CONUS). The trained model for individual reservoir has been converted to if-then statements for easy implementation with large-scale hydrological and water resources models. The inventory of the empirical operation rules is shared via [our HydroShare repository](https://doi.org/10.4211/hs.63add4d5826a4b21a6546c571bdece10).

The model is written purely in Python, with `hmmlearn` package for the hidden Markov part and `scikit-learn` package for the decision tree part. 

## 2. What's in this repository?

- `base_code` folder: modified base code files for Python `hmmlearn` package. The modification is contributed by Dr. Qiankun Zhao.
- `notebook` folder: 
  - a Jupyter notebook `GDROM` (written by Yanan Chen) for going through the model training procedure, as well as the post-process. An example reservoir (Echo Reservoir, ID: 449) is attached (`449.csv`) for you to go through the notebook.
  - a Jupyter notebook `export_rules` for exporting the trained operation rules to the if-then statements.
- `environment.yml` file that contains the dependencies of our running environment. You can directly copy to your Python environment. 

## 3. How to set up the Python environment?

### 3.1. Install all dependencies from the `environment.yml` file

A virtual environment is strongly recommended. All dependencies can be installed from the `environment.yml` by running the following command:

<!-- If you're using `conda`: 
```
conda install --file /path/to/requirements.txt
```

If you're using `pip`: 
```
pip install -r /path/to/requirements.txt
``` -->

### 3.2. Add modified base code to site packages

Since we have modified base code for `hmmlearn` package to couple hidden Markov Model and decision tree, you need to add the modified code files to the site package folder to replace the original one. 

To find the location of the package, start a Python kernal and activate the environment, then execute:
```python
import hmmlearn
print(hmmlearn.__file__)
```
Then you should be able to see the absolute path to the `__init__.py`. Go to its parent folder and paste the modified base code files there.

**Note**: the `hmmlearn-0.2.3` we used can not be installed on Mac with M1 chip. In this case, you may run Python on [Rosetta](https://support.apple.com/en-us/HT211861), or use the latest release for `hmmlearn` but at your own risk since we didn't test it with the latest version.

## 4. Major contributors and contact information

```
Donghui Li - donghui3@illinois.edu
Yanan Chen - yananc3@illinois.edu
Qiankun Zhao 
```

## 5. Other information

The full dataset containing all training data and extracted models for the 450+ reservoirs is hosted in Hydroshare ([link](https://www.hydroshare.org/resource/63add4d5826a4b21a6546c571bdece10/)). A more detailed description of the model, data preprocess, and model output can be found there.

References of related papers
- Zhao, Q., and Cai, X. (2020). Deriving representative reservoir operation rules using a hidden Markov-decision tree model. Advances in Water Resources, 146, 103753. ([link](https://doi.org/10.1016/j.advwatres.2020.103753))
- Chen, Y., Li, D., Zhao, Q., & Cai, X. (2022). Developing a generic data-driven reservoir operation model. Advances in Water Resources, 167, 104274. ([link](https://doi.org/10.1016/j.advwatres.2022.104274))
- Li, D., Chen Y., Lyu, L., & Cai, X. Operation rules and patterns for 452 large reservoirs in the Contiguous United States (Under Revision, WRR)
