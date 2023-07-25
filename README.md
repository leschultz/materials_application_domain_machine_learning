# Materials Application Domain Machine Learning (MADML)

Research with respect to application domain with a materials science emphasis is contained within. The GitHub repo can be found in [here](https://github.com/leschultz/application_domain.git).

## Examples

* Tutorial 1: Assess and fit a single model from all data: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leschultz/materials_application_domain_machine_learning/blob/main/examples/jupyter/tutorial_1.ipynb)


## Structure
The structure of the code packages is as follows:

```
materials_application_domain_machine_learning/
├── examples
│   ├── auto_push
│   │   ├── fit.py
│   │   ├── run.sh
│   │   └── submit.sh
│   └── single_runs
│       ├── make_runs.sh
│       ├── submit_jobs.sh
│       └── template
│           ├── fit.py
│           ├── run.sh
│           └── submit.sh
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.py
├── src
│   └── madml
│       ├── data
│       │   ├── citrine_thermal_conductivity_simplified.xlsx
│       │   ├── Dataset_electromigration.xlsx
│       │   ├── Dataset_Perovskite_Opband_simplified.xlsx
│       │   ├── dielectric_constant_simplified.xlsx
│       │   ├── diffusion.csv
│       │   ├── double_perovskites_gap.xlsx
│       │   ├── elastic_tensor_2015_simplified.xlsx
│       │   ├── fluence.csv
│       │   ├── heusler_magnetic_simplified.xlsx
│       │   ├── Perovskite_stability_Wei_updated_forGlenn.xlsx
│       │   ├── perovskite_workfunctions_AO_simplified.xlsx
│       │   ├── piezoelectric_tensor.xlsx
│       │   ├── steel_strength.csv
│       │   └── Supercon_data_features_selected.xlsx
│       ├── datasets.py
│       ├── hosting
│       │   └─── docker.py
│       ├── __init__.py
│       ├── ml
│       │   ├── assessment.py
│       │   ├── __init__.py
│       │   └── splitters.py
│       ├── models
│       │   ├── combine.py
│       │   ├── __init__.py
│       │   ├── space.py
│       │   └── uq.py
│       ├── plots.py
│       ├── templates
│       │   └── docker
│       │       ├── Dockerfile
│       │       ├── model_predict.py
│       │       └── user_predict.py
│       └── utils.py
└── tests
    ├── test_load_data.py
    └── test_run.py
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
