# TODOs

## Repository Cleanup
  - update default example to a dataset with binary features (e.g., mushrooms)?
  - example_00 in readme should just "pull data from URL"?
    - see e.g., https://github.com/ustunb/actionable-recourse
  - remove cvindices from repository (make sure it's saved as a version)
  - add flags for current status?
  
## Implement `check_data_for_riskslim` to check (X, y) in `RiskSLIM`Classifier`
  - X should be all finite - (n_variables x n_samples matrix)
  - y should be all finite and in 0,1 or -1,1 (flat n_samples array)
  - sample_weights should be all finite, positive (flat n_samples array)
  - make sure dimensions of X, y, and sample_weights match
  - convert y \in 0,1 to y \in -1,1 in this function?
  - output warnings if X is not binary
  - **issue warning if any column of X are constant**
     - This will rule out intercept

## RiskScores
- Add/Label Decision Points on ROC Curve and Reliability Diagram
  - "O" to Show Decision Points on ROC Curve
  - "O" with Scores
- Update so that it's calling meaningful properties (this will let us call it for e.g.., other LR models if we needed a baseline)
    - Estimator.rho <- Estimator.coefficients
    - Estimator.intercept
    - Estimator.coef_set.names <-
- Consider plopping report code into Jinja template?
  - It'll be easier to change in future.
  
## RiskSLIMClassifier
  - Check for todos in code
  - Intercept: 
    - We should assume that X doesn't contain a column of ones for the intercept
    - Let's add this ourselves
  - Add __repr__ and __str__
    - Show "fitted status"
    - Show "current score"
  - Refactor so that this is standalone object 
    - It should not "extend" RiskSLIMOptimizer, but call it in `.fit`
      ```
      self.mip = RiskSLIMOptimizer(X, y, coef_set)
      self.mip.optimize
      ```
    - This will mean adding support for e.g., _coefficients here
  - Move `create_report` into RiskScores.from_risk_score()
  
    
## RiskScoreOptimizer
  - Turn into Standalone class (rather than super class for RiskSLIM) 
    - variable_names, min_coef, max_coef <- should be passed in via coefficient_set
    - outcome_name <- optimizer doesn't need outcome_name
  - Move "initialization code" in optimize() to init() 
    - **init__ doesn't have to comply to scikit-learn (confirm?).** 
  - Optimize should be designed to be called multiple times
    - Do not warm-start if it has already been fitted once
  - Provide easy access to mip
  
  - Add _repr_ and _str_ to show info about optimizer
  - To Remove
    - ChainedUpdates
    - Timing Stats
    - ~~Support for w_pos/w_neg~~
  - Use `info` or `stats` not both
  - Plop loss function choice into factory functions?
  - Check for todos in code

## Before Release
- Test for Solver
- Installable
- Upload to PyPI?

## Publicity
- E-mail to MDCalc?
- HackerNews
- Reddit

