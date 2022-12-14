# Evaluation

[For Master Branch, not for the current branch with ongoing development]

The goal of this module is to develop a consistent method of evaluating the performance of predictions for the patient panel.  

Use:
1. First make all the necessary tweaks in `config/config.yaml`. You will have all the parameters you want to tweak for the evaluation. Everything is documented.
2. Change the `path/to/dir` of config.yaml in the Evaluation.ipynb if it is anything different from `./config/config.yaml`.
3. Comment/Uncomment the two functions for generating synthetic data inside the main function in Evaluation.ipynb based on your need. 
4. Run and see all the evaluations you wanted. 
5. You can run multiple experiments (run the whole program multiple times) and all the results will be appended. You have the option to change this in config.yaml

Directory Structure:
 - Prediction File resides in `./data/predictions/predictions.csv`
 - Referral File resides in `./data/referrals/referrals.csv`
 - Result will be saved in `./results/results.csv`
 
Data in this repository are simulated/toy data. Original data is Proprietary to CDPHP and is not released outside. 

I've followed the same directory structure for local machine and for S3. If any changes made to the directory structure, the config.yaml file should be changed accordingly.

Publication: https://ieeexplore.ieee.org/abstract/document/9669714 
