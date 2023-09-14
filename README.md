# Materials Application Domain Machine Learning (MADML)

Research with respect to application domain with a materials science emphasis is contained within. The GitHub repo can be found in [here](https://github.com/leschultz/application_domain.git).

## Examples

* Tutorial 1: Assess and fit a single model from all data: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leschultz/materials_application_domain_machine_learning/blob/main/examples/jupyter/tutorial_1.ipynb)
* Tutorial 2: Use model hosted on Docker Hub: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leschultz/materials_application_domain_machine_learning/blob/main/examples/jupyter/tutorial_2.ipynb)

## Structure
The structure of the code packages is as follows:

```
materials_application_domain_machine_learning/
├── examples
│   ├── auto_push
│   ├── jupyter
│   └── single_runs
│       ├── bw_rf
│       ├── gt_rf
│       ├── kernel_rf
│       ├── nn
│       ├── ols
│       ├── rf
│       ├── svm
│       └── wg_rf
├── src
│   └── madml
│       ├── data
│       ├── hosting
│       ├── ml
│       ├── models
│       └── templates
│           └── docker
└── tests
```

## Coding Style

Python scripts follow PEP 8 guidelines. A usefull tool to use to check a coding style is pycodestyle.

```
pycodestyle <script>
```

## Authors

### Graduate Students
* **Lane Schultz** - *Main Contributer* - [leschultz](https://github.com/leschultz)

## Acknowledgments

* The [Computational Materials Group (CMG)](https://matmodel.engr.wisc.edu/) at the University of Wisconsin - Madison
* Professor Dane Morgan [ddmorgan](https://github.com/ddmorgan) and Dr. Ryan Jacobs [rjacobs914](https://github.com/rjacobs914) for computational material science guidence
