# Evaluation

[For Master Branch, not for the current branch with ongoing development]

The goal of this module is to develop a consistent method of evaluating the performance of predictions for the patient panel.  

I've taken a small dataset of a few referrals and non referrals, and I've generated a set of test prediction datasets that should get them all correct. They are found under:

For example, the file: 

`/data/predictions/tests/tests_100_shift-12.csv`

Should have all the correct predictions (at probability 1.00) just shifted ahead by 12 months. 

`/data/predictions/tests/tests_100_shift0.csv` is a 100% correct in the correct month. 

The other set of files: 
`/data/predictions/tests/tests_75_shift0.csv`

Are identical except they have .75 as max prediction rather than 1.00.

