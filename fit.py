
# This module gives easy access to fitting functions and supports
# the quantities module.

# Known issue, to be investigated:
#   In test_linear the boostrap algorithm almost always finds a
#   too shallow slope, whereas ODR gives a better result.
#   It matches curve_fit for y errors only.

import inspect
import numpy as np
import numpy.random as random
import quantities as pq

import scipy.optimize as opt
import scipy.odr as odr
import scipy.stats as stats

from fmt import *

# see http://stackoverflow.com/a/21844726
def bootstrap(fun, p0, xdata, ydata,
              xerr=None, yerr=None,
              runs=1000, pprint=''):
    """Estimate parameters and their standard deviation from
      given data with uncertainty using boostrap Monte-Carlo method.
      Works on pure numbers without units.

      When xerr and yerr are None, scipy's leastsq is used for an initial fit.
      The standard deviation of residuals is then used to scatter the ydata
      points for boostrapping.

      Arguments:
        fun: The model function `ys = f(xs, *params)`
        p0: Initial parameters.
        runs: The number of data sets to generate
        pprint: When bool(pprint) is True some explaining output is generated
          and printed to stdout prepended with pprint (so you can use it
          to add indentation).

      Returns: `(params, std)`
        Where params is a list of fitted parameters and std their
        standard deviation.

      Raises:
        RuntimeError: If a leastsq run didn't find a solution.
    """
    if pprint: print(pprint+"Bootstrapping with %d runs"%runs)
    errfunc = lambda p, x, y: fun(x, *p) - y
    if xerr is None and yerr is None:
        if pprint: print(pprint+"    Estimating std dev of residuals")
        params, pcov, info, msg, ier =\
            opt.leastsq(errfunc, p0, args=(xdata, ydata),
                        full_output=1)
        if not (1 <= ier <= 4):
            raise RuntimeError("Could not estimate residuals: %s"%msg)
        s_res = np.std(errfunc(params, xdata, ydata))
        if pprint:
            print(pprint+"        parameters: "+repr(param))
            print(pprint+"        std dev of residuals: %f"+s_res)
    ps = []
    for i in range(runs):
        if xerr is None and yerr is None:
            xs = xdata
            ys = np.random.normal(ydata, s_res)
        else:
            if xerr is not None: xs = random.normal(xdata, xerr)
            else: xs = xdata
            if yerr is not None: ys = random.normal(ydata, yerr)
            else: ys = ydata
        rparam, rcov, info, msg, ier =\
            opt.leastsq(errfunc, p0, args=(xs, ys),
                        full_output=1)
        if not (1 <= ier <= 4):
            raise RuntimeError("No solution found for all data sets: %s"%s)
        ps.append(rparam)
    ps = np.array(ps)
    params = np.mean(ps, 0)
    std = np.std(ps, 0)
    if pprint:
        print(pprint+"    parameters: "+repr(params))
        print(pprint+"    std dev: "+repr(std))
    return (params, std)

def fit(fun, xdata, ydata, p0,
        xerr=None, yerr=None,
        usebootstrap=False, pprint=True):
    """This is a helper function dealing with the quantities and is handling
      over the real math to other fit functions.

      Arguments:
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
        usebootstrap: Use bootstrapping instead of some scipy fitting algorithm.
            Bootstrapping works for all types of uncertainties.
        pprint: Set True to see some more information about fit.

      Returns:
        params: A list of parameters as pq.Quantity
        std: Standard deviations of parameters in same units
        info: A dict with the following keys:
          'method': 'curve_fit' or 'odr'
          'ok': True when the fit was probably successful
          'cov' (optional): the covariance matrix
          'res_var': residual variance / reduced chi squared
          'msg' (optional): some information about fit in english
    """
    assert len(np.shape(xdata)) is 1
    assert len(np.shape(ydata)) is 1
    n = len(xdata)
    assert n == len(ydata)
    dof = n - len(p0)
    assert dof > 0
    # inspect model function for parameter names
    argspec = inspect.getargspec(fun)
    paramnames = None
    if argspec.varargs is None:
        if len(p0) != len(argspec.args) - 1:
            raise ValueError("Number of arguments of model function"
                             " is not matching length of p0.")
        paramnames = argspec.args[1:]
    else:
        paramnames = ["param%d"%(i+1) for i in range(dof)]
    # strip units
    assert isinstance(xdata, np.ndarray)
    assert isinstance(ydata, np.ndarray)
    if isinstance(xdata, pq.Quantity):
        xunits = xdata.units
        xdata = xdata.magnitude
    else: xunits = pq.dimensionless
    if isinstance(ydata, pq.Quantity):
        yunits = ydata.units
        ydata = ydata.magnitude
    else: yunits = pq.dimensionless
    if xerr is not None:
        if isinstance(xerr, pq.Quantity):
            xerr = xerr.rescale(xunits).magnitude
        assert len(np.shape(xerr)) == 0 or np.shape(xerr) == (len(xdata))
        if np.amin(xerr) <= 0:
            raise ValueError("Zeros / neg values in x-uncert not allowed")
    if yerr is not None:
        if isinstance(yerr, pq.Quantity):
            yerr = yerr.rescale(yunits).magnitude
        assert len(np.shape(yerr)) == 0 or np.shape(yerr) == (len(ydata))
        if np.amin(yerr) <= 0:
            raise ValueError("Zeros / neg values in y-uncert not allowed")
    p0 = [p if isinstance(p, pq.Quantity) else p * pq.dimensionless
                for p in p0]
    p0units = [p.units for p in p0]
    p0 = [p.magnitude for p in p0]
    # wrap function appending units again
    def func(x, *params):
        params2 = [p*u for p,u in zip(params, p0units)]
        return fun(x * xunits, *params2).rescale(yunits).magnitude

    # fit
    if pprint:
        print("Fitting %s depending on %s"%(repr(yunits)[13:],
                                            repr(xunits)[13:]))
        print("    N=%d, params=%d, dof=%d"%(n, len(p0), dof))
    params = None # estimated parameters
    std = None    # standard deviation
    info = {}     # additional information
    if usebootstrap:
        if pprint: print("    Method: bootstrap")
        info['method'] = 'bootstrap'
        try:
            if pprint: ppprint = '    '
            else: ppprint = False
            params, std = bootstrap(fun, p0, xdata, ydata, xerr, yerr,
                                    pprint=ppprint)
        except RuntimeError as e:
            info['ok'] = False
            info['msg'] = str(e)
        else:
            info['ok'] = True
            if xerr is None and yerr is None:
                info['res_var'] = np.sum((func(xdata, *params)-ydata)**2) / dof
            elif xerr is None:
                info['res_var'] = (np.sum((func(xdata, *params)-ydata)**2
                                         / yerr**2) / dof)
    elif xerr is None and yerr is None:
        if pprint: print("    Method: scipy leastsq")
        info['method'] = 'leastsq'
        errfunc = lambda p, x, y: fun(x, *p) - y
        params, pcov, inf, msg, ier =\
            opt.leastsq(errfunc, p0,
                        args=(xdata, ydata),
                        full_output=1)
        if not (1 <= ier <= 4):
            info['ok'] = False
            info['msg'] = msg
        else:
            info['ok'] = pcov is not None
            info['res_var'] = np.sum((func(xdata, *params)-ydata)**2) / dof
            info['msg'] = msg
            if pcov is not None:
                # multiply by residual variance because leastsq
                # returns the reduced covariance matrix
                info['cov'] = pcov * info['res_var']
                std = [np.sqrt(pcov[i][i]) for i in range(len(p0))]
            else: std = [np.inf] * len(p0)
    elif xerr is None:
        if pprint: print("    Method: scipy curve_fit")
        info['method'] = 'curve_fit'
        try:
            params, pcov, inf, msg, ier =\
                opt.curve_fit(func, xdata, ydata, p0,
                            sigma=yerr, #absolute_sigma=True,
                            full_output=1)
        except RuntimeError as e:
            info['ok'] = False
            info['msg'] = str(e)
        else:
            info['ok'] = not np.any(np.isinf(pcov))
            info['msg'] = msg
            info['cov'] = pcov
            info['res_var'] = (np.sum((func(xdata, *params)-ydata)**2 / yerr**2)
                               / dof)
            std = [np.sqrt(pcov[i][i]) for i in range(len(p0))]
    else:
        if pprint: print("    Method: scipy ODR")
        info['method'] = 'odr'
        # change function signature again
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
        params = out.beta
        std = out.sd_beta

    # append units
    params = [p*u for p,u in zip(params, p0units)]
    std = [s*u for s,u in zip(std, p0units)]
    # pretty print
    if pprint:
        print("    successful: %s"%info['ok'])
        if 'msg' in info:
            print("   ", info['msg'])
        if 'res_var' in info:
            print("    Residual variance: %f"%info['res_var'])
            p = stats.chi2.cdf(info['res_var']*dof, dof)
            print("    Probability of fit: {:.1f}%".format((1-p)*100))
        if params and std:
            print("    Parameters:")
            for n,p,s in zip(paramnames, params, std):
                print("      %s: %s"%(n, fmtquant(p, s, unit=True)))
    return params, std, info


def distribute(xs, xerr):
    """Add normal distributed error to all values in `xs`
      according to `xerr`.  Rescale xerr to units of xs.
    """
    if isinstance(xs, pq.Quantity):
        unit = xs.units
        xs = xs.magnitude
        assert isinstance(xerr, pq.Quantity)
        xerr = xerr.rescale(unit).magnitude
    else: unit = 1
    dist = random.normal(xs, xerr)
    return dist * unit

def draw_samples(fun, xs, params, xerr=None, yerr=None):
    """Calculate function values using `params` and then distribute
      `xs` and `ys` according to `xerr` and `yerr`.

      Returns: (xdata, ydata)
    """
    ys = fun(xs, *params)
    if xerr: xs = distribute(xs, xerr)
    if yerr: ys = distribute(ys, yerr)
    return (xs, ys)

def test_model(fun, p0, xs, xerr, yerr):
    xdata, ydata = draw_samples(fun, xs, p0, xerr=xerr, yerr=yerr)
    pinit = [p * random.normal(1, 0.1) for p in p0]
    # fit
    p1, std1, info1 = fit(fun, xdata, ydata, pinit,
                          yerr=yerr, xerr=xerr)
    # bootstrap
    p2, std2, info2 = fit(fun, xdata, ydata, pinit,
                          yerr=yerr, xerr=xerr,
                          usebootstrap=True)
    # compare
    print("Comparison: %s <-> %s (showing deviation from model"
          " parameter)"%(info1['method'], info2['method']))
    for i in range(len(p0)):
        print("    Parameter {:d}: {} <-> {} (model: {})".format(i,
                fmtquant(np.absolute(p1[i]-p0[i])/std1[i]),
                fmtquant(np.absolute(p2[i]-p0[i])/std2[i]),
                fmtquant(p0[i])))
    print()
    # plot
    import matplotlib.pyplot as plt
    plt.figure()
    if xerr or yerr:
        plt.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, fmt=' ', label='data')
    else:
        plt.plot(xdata, ydata, '.', label='data')
    plt.plot(xs, fun(xs, *p0), label='model')
    plt.plot(xs, fun(xs, *p1), label=info1['method'])
    plt.plot(xs, fun(xs, *p2), label=info2['method'])
    plt.legend(loc='best', numpoints=1)
    plt.show()

def test_linear():
    a = -0.5 * pq.nm / pq.s
    b = 5 * pq.nm
    def f(xs, a, b):
        return xs * a + b
    xbase = np.linspace(0, 16, 101) * pq.s
    test_model(f, (a,b), xbase, None, 1*pq.nm)
    test_model(f, (a,b), xbase,
               1/np.sqrt(2)*pq.s, 1/np.sqrt(2)*pq.nm)

def test_exp():
    a = 5 * pq.nm
    b = -0.1 / pq.s
    c = 0.5 * pq.nm
    def f(xs, a, b, c):
        return a * np.exp(b * xs) + c
    xbase = np.linspace(0, 30, 101) * pq.s
    test_model(f, (a,b,c), xbase, None, 0.5*pq.nm)
    test_model(f, (a,b,c), xbase,
               0.3 * pq.s, 0.3 * pq.nm)

def test_chi2(N=300, n=101,
              xerr=None, #0.2*pq.s,
              yerr=0.1*pq.nm):
    a = 10 * pq.nm / pq.s
    b = 5 * pq.nm
    def f(xs, a, b):
        return xs * a + b
    chi2rs = []
    print("Chi2_red distribution")
    print("    Fitting %d points %d times, ..."%(n, N))
    xbase = np.linspace(0, 50, n) * pq.s
    for i in range(N):
        xdata, ydata = draw_samples(f, xbase, (a,b), xerr=xerr, yerr=yerr)
        pinit = [p * random.normal(1, 0.1) for p in (a, b)]
        param, std, info = fit(f, xdata, ydata, pinit,
                               yerr=yerr, xerr=xerr,
                               pprint=False)
        chi2rs.append(info['res_var'])
    chi2rs = np.array(chi2rs)
    print("    Mean:",np.mean(chi2rs))
    print("    Std dev:",np.std(chi2rs))

if __name__ == '__main__':
    test_chi2(n=100)
    #test_linear()
    #test_exp()
