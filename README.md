# A flexible framework for automated STED super-resolution microscopy

**autoSTED** allows you to automate an Abberior Instruments STED microscope in a flexible and generic way.
We have sucessfully used it for multiple super-resolution studies dealing mainly with chromatin organization:
* [Brandstetter et al. (Biophysical Journal, 2022)](https://linkinghub.elsevier.com/retrieve/pii/S0006349522001096)
* [Palikyras et al. (Aging Cell, 2024)](https://onlinelibrary.wiley.com/doi/10.1111/acel.14083)
* [Steinek et al. (Cell Reports Methods, 2024)](https://doi.org/10.1016/j.crmeth.2024.100840)
* [Stumberger et al. (bioRxiv preprint, 2025)](http://biorxiv.org/lookup/doi/10.1101/2025.01.20.633941)

A technical manuscript describing the framework is currently in preparation.

## Installation

### SpecPy and compatible NumPy

Our framework talks to the microscope control software Imspector via the SpecPy interface.

SpecPy ```.whl``` files can be found in the Imspector installation folder (e.g. ```C:\Imspector\Versions\{VersionNr}\python\specpy\{SpecPy_Version_Nr}```). SpecPy is built for specific Python and NumPy versions, indicated in the file/folder name, so it is a good idea to create a conda environment with the corresponding versions (e.g. for Python 3.11.4, NumPy 1.24.3):

```bash
conda create -n autosted-env python=3.11.4 numpy=1.24.3
```

Then, in you new environment, you can install SpecPy:

```bash
conda activate autosted-env
pip install C:/Imspector/path/to/specpy.whl
```

### autoSTED

Once you have an environment set, you can install autoSTED (this repository):

```bash
git clone https://github.com/hoerlteam/sted_automation.git
cd sted_automation
pip install .
```

Alternatively, install directly without cloning:

```bash
pip install git+https://github.com/hoerlteam/sted_automation.git
```

### Aviod NumPy updates

**Warning:** Installing additional third-party packages may sometimes update NumPy, causing SpecPy to stop working. Consider using the ```--dry-run``` options of ```pip/conda``` to check if an installation might cause problems.

Sometimes, it helps to explicity re-state the NumPy version during install:

```bash
pip install package-of-choice numpy==1.24.3
conda install package-of-choice numpy=1.24.3
```

Pinning the NumPy version in ```conda``` might also be worth a look: [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning)

## Usage

You can find Jupyter notebooks showcasing the use of autoSTED under ```examples```. To run the notebooks, you should also install jupyter into your environment via conda/pip.

We assume that the microscope and Imspector are running and typically that the sample is (roughly) in focus when an automation pipeline is started.