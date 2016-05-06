
# This module gives easy access to fitting functions and supports
# the quantities module.

import inspect
import numpy as np
import numpy.random as random
import quantities as pq

import scipy.optimize as opt
import scipy.odr as odr
import scipy.stats as stats

from physics.fmt import fmtquant

class FitError(Exception):
    pass

def autofit(fun, xdata, ydata, p0,
            xerr=None, yerr=None,
            bounds=None):
    """This function selects the most specialized function from the SciPy
      module for fitting data:

      no bounds, xerr and yerr are None:
          `scipy.optimize.leastsq`
          Levenberg-Marquardt
      no bounds, xerr is None:
          `scipy.optimize.curve_fit`
          Levenberg-Marquardt
      no bounds, xerr and optional yerr given:
          `scipy.odr.odr`
          Levenberg-Marquardt
      bounds and optional yerr given:
          `scipy.optimize.minimize`
          L-BFGS-B

      Fitting data with xerr and bounds is not supported.
      Try using `inverse_fitquant()`.

      Fitting may give bad results when the parameters differ in orders
      magnitude.  In that case rescale the model function.
      afaik ODR does that automatically.

      This function does not work with python-quantities.
      Use the wrapper `fitquant()`.

      Arguments:
        fun: the model function `fun(x, param1, param2, ...)`.  It has to work
            on single values and numpy.ndarrays for x.
        xdata: the xdata as 1-dim array
        ydata: the ydata as 1-dim array
        p0: list of initial guesses for parameters.
        xerr: the uncertainty for xdata (scalar or of same length as xdata)
        yerr: the uncertainty for ydata (scalar or of same length as ydata)
        bounds: list of (min, max) tuples for parameters.  Use None for min or
            max when there is no bound in that direction.  If parameters lie
            close to a bound, the estimated covariance matrix becomes
            singular (not good).

      Returns: a 3-tuple of
        params: An array of estimated parameters
        stdev: An array of estimated standard deviations for parameters,
            may be `inf` if fit failed.
        info: dict with keys (optional keys may be undefined)
            'method': 'leastsq', 'curve_fit', 'odr' or 'minimize'
            'success': True when the fit was probably successful (converged)
            'message': some information about fit in english
            'res_var': residual variance / reduced chi-squared
            'probability': probability of fit according to
                chi-squared distribution
            'pcov' (optional): the covariance matrix as ndarray
            'out' (optional): the full output of the used algorithm

      Raises:
        FitError: When dimensions of input data are not compatible, or no
            algorithm was found for data.
    """
    # check data
    if isinstance(xdata, pq.Quantity) or isinstance(ydata, pq.Quantity):
        raise FitError("xdata or ydata is a Quantity.  Use fitquant.")
    if isinstance(xerr, pq.Quantity) or isinstance(yerr, pq.Quantity):
        raise FitError("xerr or yerr is a Quantity.  Use fitquant.")
    if len(np.shape(xdata)) != 1 or len(np.shape(ydata)) != 1:
        raise FitError("xdata and ydata have to be 1-dimensional arrays")
    if len(ydata) != len(xdata):
        raise FitError("xdata and ydata are not of same length")
    if xerr is not None:
        if not np.isscalar(xerr) and xerr.shape != () \
           and np.shape(xerr) != np.shape(xdata):
            raise FitError("xerr not of same shape as xdata")
        if np.min(xerr) <= 0:
            raise FitError("xerr contains data points <= 0")
    if yerr is not None:
        if not np.isscalar(yerr) and yerr.shape != () \
           and np.shape(yerr) != np.shape(ydata):
            raise FitError("yerr not of same shape as ydata")
        if np.min(yerr) <= 0:
            raise FitError("yerr contains data points <= 0")
    if bounds:
        if len(bounds) != len(p0):
            raise FitError("The number of bounds does not match the number"
                           " of parameters.")
        # rewrite bounds for curve_fit
        mins = np.array([b[0] for b in bounds])
        maxs = np.array([b[1] for b in bounds])
        bounds = mins, maxs

    n = len(xdata)    # number of data points
    dof = n - len(p0) # degrees of freedom
    params = None # estimated parameters
    std = None    # standard deviation
    info = {}     # additional information

    if dof <= 0:
        raise FitError("degrees of freedem <= 0")

    if xerr is None:
        info['method'] = 'curve_fit'
        kwargs = {}
        if bounds:
            kwargs['bounds'] = bounds
        else:
            kwargs['full_output'] = True
        try:
            res = opt.curve_fit(fun, xdata, ydata, p0,
                                sigma=yerr, absolute_sigma=True,
                                **kwargs)
        except RuntimeError as e:
            info['success'] = False
            info['message'] = str(e)
        else:
            if len(res) == 2:
                params, pcov = res
                info['message'] = 'trf finished'
            else:
                params, pcov, fullout, msg, ier = res
                info['out'] = fullout
                info['message'] = msg
            info['success'] = not np.any(np.isinf(pcov))
            info['res_var'] = (np.sum((ydata-fun(xdata, *params))**2
                                      / yerr**2) / dof)
            info['pcov'] = pcov
            std = np.array([np.sqrt(pcov[i][i]) for i in range(len(p0))])
    elif bounds is None: # xerr and optional yerr
        info['method'] = 'odr'
        # change function signature
        def odrf(B, x):
            return fun(x, *B)
        # run ODR
        model = odr.Model(odrf)
        data = odr.RealData(xdata, ydata, sx=xerr, sy=yerr)
        regr = odr.ODR(data, model, beta0=p0)
        out = regr.run()
        # evaluate
        info['success'] = (1 <= out.info <= 3)
        info['message'] = out.stopreason
        info['res_var'] = out.res_var
        info['pcov'] = out.cov_beta
        info['out'] = out
        params = out.beta
        std = out.sd_beta
    else:
        raise FitError("Fitting data with xerr and bounds not supported."
                       " Try inverse_fitquant().")
    if 'res_var' in info:
        info['probability'] = 1 - stats.chi2.cdf(info['res_var']*dof, dof)
    return params, std, info


def fitquant(fun, xdata, ydata, p0,
             xerr=None, yerr=None, bounds=None,
             pprint=True, full_output=False):
    """This is a wrapper function for `autofit()` dealing with quantities.

      Arguments:
        fun: The model function `f(x, param1, param2, ...)`
        xdata: the x data points as Quantity or simple ndarray (dimensionless).
        ydata: the y data points as Quantity or simple ndarray (dimensionless).
        p0: list of initial parameters.  It is used to determine number of free
            parameters.  The model function will receive the parameters in the
            same units, and results will be displayed in these units too.
        xerr: Uncertainty for xdata, either a single value for all points, or
            an array of same length as xdata.  Will be rescaled to same unit as
            xdata before fitting.
        yerr: Uncertainty for ydata.  See xerr.
        bounds: Boundaries for parameters are scaled to same unit as
            corresponding initial parameters.
        pprint: If True, fitting results are printed to stdout.
        full_output: Set True to get 3-tuple instead of 2-tuple as return value
            containing the info dict as third element.

      Returns:
        params: list of parameters as pq.Quantity
        stddev: standard deviations of parameters in same units
        infodict (only with full_output=True): same as in `autofit()`

      Raises:
        FitError: see `autofit()`
        ValueError: if rescaling units fails
    """
    n = len(xdata)
    dof = n - len(p0)
    # strip units from data
    if isinstance(xdata, pq.Quantity):
        xunits = xdata.units
        xdata = xdata.magnitude
    else:
        xunits = 1 * pq.dimensionless
        xdata = np.array(xdata)
    if isinstance(ydata, pq.Quantity):
        yunits = ydata.units
        ydata = ydata.magnitude
    else:
        yunits = 1 * pq.dimensionless
        ydata = np.array(ydata)
    if xerr is not None and isinstance(xerr, pq.Quantity):
        try: xerr = xerr.rescale(xunits).magnitude
        except ValueError as e:
            raise ValueError("Units of xerr not matching xdata: "+str(e))
    if yerr is not None and isinstance(yerr, pq.Quantity):
        try: yerr = yerr.rescale(yunits).magnitude
        except ValueError as e:
            raise ValueError("Units of yerr not matching ydata: "+str(e))
    # params
    p0 = [p if isinstance(p, pq.Quantity) else p * pq.dimensionless for p in p0]
    p0units = list([p.units for p in p0])
    p0 = list([p.magnitude for p in p0])
    if bounds:
        if len(bounds) != len(p0):
            raise FitError("The number of bounds does not match the number"
                           " of parameters.")
        bounds2 = []
        for i, (units, (min, max)) in enumerate(zip(p0units, bounds)):
            if min is not None:
                if not isinstance(min, pq.Quantity):
                    min *= pq.dimensionless
                try: min = min.rescale(units).magnitude
                except ValueError as e:
                    raise ValueError("Units of %d. lower bound not matching"
                                     " parameter: "%(i+1)+str(e))
            if max is not None:
                if not isinstance(max, pq.Quantity):
                    max *= pq.dimensionless
                try: max = max.rescale(units).magnitude
                except ValueError as e:
                    raise ValueError("Units of %d. upper bound not matching"
                                     " parameter: "%(i+1)+str(e))
            bounds2.append( (min, max) )
        bounds = bounds2

    # inspect model function for parameter names
    argspec = inspect.getargspec(fun)
    if argspec.varargs is None:
        if len(p0) != len(argspec.args) - 1:
            raise FitError("Number of arguments of model function"
                           " not matching length of p0.")
        paramnames = argspec.args[1:]
    else:
        paramnames = ["param%d"%(i+1) for i in range(len(p0))]
    # wrap function appending units
    def func(x, *params):
        params2 = [p*u for p,u in zip(params, p0units)]
        return fun(x * xunits, *params2).rescale(yunits).magnitude

    if pprint:
        print("Fitting [%s] depending on [%s]"%(repr(yunits)[13:],
                                                repr(xunits)[13:]))
        print("    N=%d, params=%d, dof=%d"%(n, len(p0), dof))
    params, std, info = autofit(func, xdata, ydata, p0, xerr, yerr, bounds)
    params = [p*u for p,u in zip(params, p0units)]
    std = [s*u for s,u in zip(std, p0units)]
    if pprint: # pretty print
        print("    Method:", info['method'])
        print("    successful:", info['success'])
        if not info['success']:
            print("    The fit was NOT SUCCESSFUL, following data is"
                  " not optimal!")
        if 'message' in info:
            print("   ", info['message'])
        if 'res_var' in info:
            print("    Residual variance:", info['res_var'])
            print("    Probability of fit:"
                  " {:.1f}%".format(info['probability']*100))
        if params and std:
            print("    Parameters:")
            for n,p,s in zip(paramnames, params, std):
                print("      {:s}: {:s}".format(n, fmtquant(p, s)))
    if full_output:
        return params, std, info
    else:
        return params, std
