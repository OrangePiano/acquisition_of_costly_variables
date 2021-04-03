# Acqusition of Costly Information in Data-driven Decision Making
The code has been developed as a part of master thesis Acquisition of Costly Information in Data-Driven Decision Making at Institute of Economic Studies, Charles University. The code provides several methods for acquisition of costly variables utilizing supervised machine learning models.

## Software requirements
The code is written in Python 3.8.3 and uses the following external libraries:

```
numpy 1.19.1
pandas 1.1.2
pickle 4.0
scikit-learn 0.23.2
seaborn 0.11.1
```

## File list
- 'decision_maker.py' - the main script conducting the acquisition of costly variables
- 'experiments.ipynb' - a notebook running the experiments on euthyroid data (original source: http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/) and NHANES diabetes data (original source: https://github.com/mkachuee/Opportunistic)
- 'euthyroid.pkl' - cleaned euthyroid data, contains features, their costs and targets

## Usage
For the usage see the 'experiments.ipynb' notebook. It provides examples of how to run the 'decision_maker.py'. See also documentation in the 'decision_maker.py' for further details about the acqusition of costly variables.

The 'decision_maker.py' currently does not support:
- costly categorical variables
- other than classification tasks
