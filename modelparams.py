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

INTERCEPT = 7.5980 
HPA_COEF = -0.0811
TXATABOVE_COEF = 0.0003 
INTERACT_COEF = 0 
HIGHGAS_COEF = 0.4085
#high gas offered is defined based as a percentage of the gas limit
#highgas2 is the only one that matters right now

HIGHGAS1 = .037
HIGHGAS2 = .15

