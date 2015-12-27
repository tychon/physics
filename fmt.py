
import io
import math
import numpy as np
import quantities as pq

import subprocess
import os

logten = math.log(10.0)
# TODO: handle infinit values and uncertainties
def fmtuncert(value, uncert=None,
              decimals=None, power=None, significance=None,
              pm=' +- ', ten=' 10^', tex=False, paren=False,
              nanempty=True):
    """Format floating point values with uncertainty.
      Does not work with numpy arrays.
      Be aware of the rounding problem for the binary representation of values.

      Arguments:
          value, uncert: The floating point value and its uncertainty.
              uncert may be None, then the plus-minus is omitted.
          decimals: The number of digits to print after the decimal point.
              When None it is set so that two significant digits of uncert
              are visible, or four significant digits if uncert is zero or None.
          power: The power to display the number in.  When non-zero the number
              and its uncertainty are surrounded by parentheses.
          significance: The value and error are rounded to this power of ten
              before formatting (but after estimating power / decimals)
          pm: The separator between the value and its uncertainty.
          ten: The separator between the uncertainty (in parentheses) and
              the power.
          tex: If tex is True the power is surrounded by curly brackets.
          paren: If True, surround text with parentheses when uncertainty
              is shown, even if power is zero.
          nanempty: If True, an empty string is returned when value or
              uncert is np.nan

      Returns:
          A formatted string.
    """
    if nanempty and (np.isnan(value) or
                     uncert is not None and np.isnan(uncert)):
        return ""
    if power is None:
        if value == 0.0: power = 0
        else: power = int(math.floor(math.log(abs(value))/logten))
        if -2 <= power <= 3: power = 0
    if decimals is None:
        if not uncert:
            # four significant digits
            if value == 0.0: decimals = 1
            else:
                decimals = -int(math.floor(math.log(abs(value / 10**power))
                                           / logten))
                decimals = max(0, decimals+3)
        else:
            decimals = -int(math.floor(math.log(uncert / 10**power)/logten))
            decimals = max(0, decimals+1)
            if round(uncert / 10**power * 10**decimals) >= 100.0:
                decimals -= 1
    decimals = max(0, decimals)
    if significance is not None:
        s = 10**significance
        value = round(value / s) * s
        if uncert is not None:
            uncert = round(uncert / s) * s
    if uncert is None:
        num = "{0:.{1}f}".format(value / 10**power, decimals)
    else:
        num = "{0:.{2}f}{3}{1:.{2}f}".format(
            value / 10**power,
            uncert / 10**power,
            decimals, pm)
        if power != 0 or paren:
            num = "("+num+")"
    if power != 0:
        if tex: fmt = "{}{}{{{:d}}}"
        else: fmt = "{}{}{:d}"
        num = fmt.format(num, ten, power)
    return num


def fmtquant(quant, *args, **kwargs):
    """Give python Quantity object (or UncertainQuantity) and any
      arguments that fmtuncert takes.

      Additional arguments:
        unit: Set True to append unit (will make parentheses around
          values).  Formats as tex formula when `tex=True` is set.
          Defaults to True.
    """
    tex = kwargs.get('tex', False)
    unit = kwargs.pop('unit', True)
    # value
    quant = 1 * quant
    value = quant.magnitude
    # uncert
    uncert = None
    if len(args) > 0:
        uncert = args[0]
        args = args[1:]
    elif 'uncert' in kwargs:
        uncert = kwargs.pop('uncert')
    if uncert is not None:
        if isinstance(uncert, pq.UncertainQuantity):
            raise ValueError("Cannot format UncertainQuantity"
                             " as uncertainty.")
        if isinstance(uncert, pq.Quantity):
            uncert = uncert.rescale(quant.units).magnitude
    elif hasattr(quant, 'uncertainty'):
        uncert = quant.uncertainty
    else:
        uncert = None
    # fmt
    kwargs.pop('paren', False)
    s = fmtuncert(value, uncert, *args, paren=True, **kwargs)
    if unit and quant.dimensionality != pq.dimensionless:
        if tex: s = s + ' ' + quant.dimensionality.latex[1:-1]
        else: s = s + ' ' + repr(quant.units)[13:]
    return s

def printtable(columns, caption="", tableno=1, filename=None,
                columnformat=None, index=[]):
    """Shorthand for formatting and printing table.
      `filename` defaults to 'tableTABLENO.pdf'.
    """
    tab = fmttable(columns, caption, tableno, columnsformat, index)
    if name is None: name = "table{}".format(tableno)
    printtex(filename, tab)

def fmttable(columns, tableno=1, caption="",
             columnformat=None, index=[],
             nanempty=True):
    """Format data as tex string threeparttable.

      Arguments:
        columns: [
            (heading, values, uncert, decimals, power=0, significance=None),
            (heading, values, fun=None)
          ]
        tableno: Numbering of table, defaults to 1.
        caption: Caption typesetted below table.
        columnformat:  List of 'r', 'c' or 'l' giving the text alignment
          in the table cells for every column.  Don't forget alignment of
          index column if you didn't set it to None.  Defaults to all
          right aligned.
        index: The index put into the first column.  Has to have the same length
          as the longest column.  Defaults to an enumeration (when set to empty
          list).  No index is printed when `index is None`.
        nanempty: Passed on to fmtuncert(), defaults to True.

      Returns: a string.
    """
    coln = len(columns) # number of cols excluding index
    colN = coln+1 if index is not None else coln # and including index
    rown = max(len(col[1]) for col in columns)
    # create enumerating index or check given one
    if index == []: index = range(1, rown+1)
    assert index is None or len(index) == rown
    # create right aligned column format or check given one
    if not columnformat:
        columnformat = 'r' * (colN)
    assert len(columnformat) == colN

    # format cells to strings
    fmt = np.empty( (rown, coln), dtype=str)
    for coli, col in enumerate(columns):
        heading = col[0]
        if 2 <= len(col) <= 3: # object / string values
            fun = col[2] if len(col) == 3 else repr
            for rowi, dat in enumerate(col[1]):
                fmt[rowi, coli] = fun(dat)
        elif 4 <= len(col) <= 6: # number values
            values = col[1]
            uncert = col[2]
            decimals = col[3]
            power = col[4] if len(col) >= 5 else 0
            signific = col[5] if len(col) >= 6 else None
            if isinstance(values, pq.Quantity):
                assert isinstance(uncert, pq.Quantity)
                if uncert.units != values.units:
                    print("INFO: rescaling uncertainty for", heading)
                    uncert = uncert.rescale(values.units)
                uncert = uncert.magnitude
                values = values.magnitude
            else:
                assert not isinstance(uncert, pq.Quantity)
            if isinstance(uncert, np.ndarray):
                if uncert.size == 1:
                    print("INFO: broadcasting uncertainty for", heading)
                    uncert = np.array([np.asscalar(uncert)] * len(values))
                if uncert.size != len(values):
                    raise ValueError("Dimension mismatch for %s"%heading)
            elif not isinstance(uncert, list):
                print("INFO: broadcasting uncertainty for", heading)
                uncert = np.array([uncert] * len(values))
            elif len(uncert) != len(values):
                raise ValueError("Dimension mismatch for %s"%heading)
            for rowi, (v, u) in enumerate(zip(value, uncert)):
                fmt[rowi][coli] = '$' + fmtuncert(v, u, decimals,
                                            power, signific,
                                            pm=r"\pm", tex=True,
                                            nanempty=nanempty) + '$'
        else:
            raise ValueError("Bad tuple for column %d"%(coli+1))

    # build string
    NL = '\n'
    s = StringIO()
    s.write(r"""
\setcounter{table}{%d}
\begin{table}
\centering
\begin{threeparttable}
\begin{tabular}{%s}
\toprule
"""%(tableno-1, columnformat))
    # header
    headings = [a[0] for a in columns]
    f.write("{} &" + " & ".join(headings) + r"\\" + NL)
    # data
    for rowi in range(rown):
        if index is not None:
            s.write(repr(index[row]))
        for coli in range(coln):
            s.write(" & ")
            s.write(fmt[rowi][coli])
        s.write(r" \\" + NL)
    # outro
    caption = r"\caption{%s}"%caption if caption else ""
    s.write(r"""\bottomrule
\end{tabular}
%s
\end{threeparttable}
\end{table}
"""%(caption))
    return s.getvalue()


def printtex(name, tex):
    """Put tex into file surrounded by some document definitions.
      Then compile it using pdflatex.  Deletes intermediary files
      when compilation was successful.

      Arguments:
        name: Basename of file to write.
          Files with extensions `.tex, .log, .aux, .pdf` are
          created / used.
        tex: The tex as string.

      Returns: True if compilation completed successfully.
    """
    texfile = name + '.tex'
    pdffile = name + '.pdf'
    # write file
    with open(texfile, 'w+') as f:
        NL = '\n'
        f.write(r"""
\documentclass[a4paper]{scrartcl}
\usepackage{microtype, lmodern, ngerman}
\usepackage{amsmath, esvect, booktabs, threeparttable}
\begin{document}"""+NL)
        f.write(tex)
        f.write(NL+r"\end{document}"+NL)

    # compile tex
    print("Compiling to "+pdffile)
    proc = subprocess.Popen(['pdflatex', texfile])
    proc.communicate()
    res = proc.returncode
    if res != 0:
        print("Compilation failed, return code: %d"%res)
        return False
    else:
        os.unlink(filename+'.log')
        os.unlink(filename+'.tex')
        os.unlink(filename+'.aux')
        return True
