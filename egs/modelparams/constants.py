'''
this file is for the constants used in the prediction model
these are generated from a geth node with globalslots set to 8000.
If you run locally, you should check to see if these are accurate using model_gas.py
you should also refit the model periodically as they may change over time.
You should have about >20k transactions in the database to fit the model accurately
'''
#intercept from poisson model
#hashpower accepting coefficient
#transactions at or above in txpool coefficient
#interaction term with highgas offered and hashpower. not currently using
#highgas offefred coefficient

INTERCEPT = 4.4683
HPA_COEF = -0.0136
TXATABOVE_COEF = .0009
AVGDIFF_COEF = -2.0235

INTERCEPT2 = 6.9238
HPA_COEF2 = -0.0670

# Highest gas price tracked
MAX_GP = 5000

