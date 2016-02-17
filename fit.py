
# This module gives easy access to fitting functions and supports
# the quantities module.

import warnings
import inspect
import numpy as np
import numpy.random as random
import quantities as pq

import scipy.optimize as opt
import scipy.odr as odr
import scipy.stats as stats

from martha.fmt import fmtquant

# show warnings every time
class FitWarning(Warning):
    pass

warnings.simplefilter('always', FitWarning)


def autofit(fun, xdata, ydata, p0,
            xerr=None, yerr=None):
    """This function selects one of the SciPy fit functions leastsq, curve_fit
      or ODR depending on given uncertainties.  This function does not work with
      Quantities.

      xerr and yerr are None: use `scipy.optimize.leastsq`,
      xerr is None: use `scipy.optimize.curve_fit`,
      both given: use `scipy.odr.odr`

      Arguments:
        fun: the model function `fun(x, param1, param2, ...)`.  It has to work
            on single values and numpy.ndarrays for x.
        xdata: the xdata as 1-dim array
        ydata: the ydata as 1-dim array
        p0: The initial guess for parameters.
        xerr: the uncertainty for xdata (scalar or of same length as xdata)
        yerr: the uncertainty for ydata (scalar or of same length as ydata)

      Returns: a 3-tuple of
        params: An array of estimated parameters
        stdev: An array of estimated standard deviations for parameters,
            may be `inf` if fit failed.
        infodict: dict with keys (optional keys may be undefined)
          'method': 'leastsq', 'curve_fit' or 'odr'
          'ok': True when the fit was probably successful (converged)
          'res_var': residual variance / reduced chi squared
          'cov' (optional): the covariance matrix
          'msg' (optional): some information about fit in english
          'out' (optional): the full output of the used algorithm
    """
    # check data
    if isinstance(xdata, pq.Quantity) or isinstance(ydata, pq.Quantity):
        warnings.warn("xdata or ydata is a Quantity.  Use fitquant.",
                      category=FitWarning)
    if isinstance(xerr, pq.Quantity) or isinstance(yerr, pq.Quantity):
        warnings.warn("xerr or yerr is a Quantity.  Use fitquant.",
                      category=FitWarning)
    if len(np.shape(xdata)) != 1 or len(np.shape(ydata)) != 1:
        raise ValueError("xdata and ydata have to be 1-dimensional arrays")
    if len(ydata) != len(xdata):
        raise ValueError("xdata and ydata not of same length")
    if xerr is not None:
        if not np.isscalar(xerr):
            if np.shape(xerr) != np.shape(xdata):
                raise ValueError("xerr not of same shape as xdata")
        if np.min(xerr) <= 0:
            warnings.warn("xerr contains data points <= 0",
                          category=FitWarning)
    if yerr is not None and not np.isscalar(yerr):
        if np.shape(yerr) != np.shape(ydata):
            raise ValueError("yerr not of same shape as ydata")
        if np.min(yerr) <= 0:
            warnings.warn("yerr contains data points <= 0",
                          category=FitWarning)

    n = len(xdata)    # number of data points
    dof = n - len(p0) # degrees of freedom
    params = None # estimated parameters
    std = None    # standard deviation
    info = {}     # additional information

    # fit
    if xerr is None and yerr is None:
        info['method'] = 'leastsq'
        errfunc = lambda p, x, y: fun(x, *p) - y
        params, pcov, fullout, msg, ier =\
            opt.leastsq(errfunc, p0,
                        args=(xdata, ydata),
                        full_output=1)
        info['out'] = fullout
        if not (1 <= ier <= 4):
            info['ok'] = False
            info['msg'] = msg
        else:
            info['ok'] = pcov is not None
            info['res_var'] = np.sum((fun(xdata, *params)-ydata)**2) / dof
            info['msg'] = msg
            if pcov is not None:
                # multiply by residual variance because leastsq
                # returns the reduced covariance matrix
                info['cov'] = pcov * info['res_var']
                std = [np.sqrt(pcov[i][i]) for i in range(len(p0))]
            else: std = [np.inf] * len(p0)
    elif xerr is None:
        info['method'] = 'curve_fit'
        try:
            params, pcov, fullout, msg, ier =\
                opt.curve_fit(fun, xdata, ydata, p0,
                            sigma=yerr, #absolute_sigma=True,
                            full_output=1)
        except RuntimeError as e:
            info['ok'] = False
            info['msg'] = str(e)
        else:
            info['out'] = fullout
            info['ok'] = not np.any(np.isinf(pcov))
            info['msg'] = msg
            info['cov'] = pcov
            info['res_var'] = (np.sum((fun(xdata, *params)-ydata)**2 / yerr**2)
                               / dof)
            std = [np.sqrt(pcov[i][i]) for i in range(len(p0))]
    else:
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
        info['ok'] = (1 <= out.info <= 3)
        info['cov'] = out.cov_beta
        info['res_var'] = out.res_var
        info['msg'] = out.stopreason
        info['out'] = out
        params = out.beta
        std = out.sd_beta
    return params, std, info


def fitquant(fun, xdata, ydata, p0,
             xerr=None, yerr=None,
             pprint=True, full_output=False):
    """This is a helper function dealing with the quantities and then
      calling `autofit`.

      Arguments:
        fun: A function like `f(x, param1, param2, ...)` returning `y`.  It
            must be able to work with `np.ndarray`s as input for x.
        xdata: The x data points as Quantity or simple ndarray (dimensionless).
        ydata: The y data points as Quantity or simple ndarray (dimensionless),
            Has to have same length as xdata.
        p0: Initial parameters, used to determine number of free parameters when
            present.  The model function will receive the paramesters in the
            same units, and results will be displayed in these units.
        xerr: Uncertainty for xdata, either a single value for all points, or
            an array of same length as xdata.  Will be rescaled to same unit as
            xdata before fitting.
        yerr: Uncertainty for ydata.
        pprint: Set True to see some more information about fit.
        full_output: Set True to get 3-tuple instead of 2-tuple as return value
            containing the infodict as third element

      Returns:
        params: A list of parameters as pq.Quantity
        stddev: Standard deviations of parameters in same units
        infodict (only with full_output=True): same as in autofit
    """
    n = len(xdata)
    dof = n - len(p0)
    # strip units
    if isinstance(xdata, pq.Quantity):
        xunits = xdata.units
        xdata = xdata.magnitude
    else:
        xunits = 1 * pq.dimensionless
        ydata = np.array(xdata)
    if isinstance(ydata, pq.Quantity):
        yunits = ydata.units
        ydata = ydata.magnitude
    else:
        yunits = 1 * pq.dimensionless
        ydata = np.array(ydata)
    if xerr is not None and isinstance(xerr, pq.Quantity):
        try: xerr = xerr.rescale(xunits).magnitude
        except ValueError as e:
            raise ValueError("Dimension of xerr not matching xdata: "+str(e))
    if yerr is not None and isinstance(yerr, pq.Quantity):
        try: yerr = yerr.rescale(yunits).magnitude
        except ValueError as e:
            raise ValueError("Dimension of yerr not matching ydata: "+str(e))
    p0 = [p if isinstance(p, pq.Quantity) else p * pq.dimensionless for p in p0]
    p0units = list([p.units for p in p0])
    p0 = list([p.magnitude for p in p0])
    # inspect model function for parameter names
    argspec = inspect.getargspec(fun)
    if argspec.varargs is None:
        if len(p0) != len(argspec.args) - 1:
            raise ValueError("Number of arguments of model function"
                             " is not matching length of p0.")
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
    params, std, info = autofit(func, xdata, ydata, p0, xerr, yerr)
    params = [p*u for p,u in zip(params, p0units)]
    std = [s*u for s,u in zip(std, p0units)]
    if pprint: # pretty print
        print("    Method:", info['method'])
        print("    successful:", info['ok'])
        if 'msg' in info:
            print("   ", info['msg'])
        if 'res_var' in info:
            print("    Residual variance:", info['res_var'])
            p = stats.chi2.cdf(info['res_var']*dof, dof)
            print("    Probability of fit: {:.1f}%".format((1-p)*100))
        if params and std:
            print("    Parameters:")
            for n,p,s in zip(paramnames, params, std):
                print("      {:s}: {:s}".format(n, fmtquant(p, s, unit=True)))
    if full_output:
        return params, std, info
    else:
        return params, std
