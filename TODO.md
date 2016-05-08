
* attribute authors of datacursors

* does quantities support ndarrays in UncertainQuantity?

* write tests

## fmt

* consistent handling of 10000 and 9999.99 (automatic scaling of
  power)

* warn when value or uncert smaller machine epsilon

* align plus-minus sign in tables when uncerts have different lengths
  (pad with zeros)

* make language configurable in table caption ('Tabelle 1')

* vectorized version of fmtquant

* plotting convenience function with DIN A4 size

* helper function for plotting fit results?

## fit

* inverse_fitquant

* add bootstrap method for determining uncertainties (bootstrapping
  many times does not make p0err smaller but more
  precise. bootstrapping with given `\Delta y_i` should only remove
  errors from estimating the jacobian, not errors from wrong
  `\Delta y_i`.)

* extra python file for test functions

* support passing a function for the Jacobian
* support passing a tolerance for termination
