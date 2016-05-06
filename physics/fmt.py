
__all__ = ['fmtuncert', 'fmtquant', 'fmtquant_vec',
           'comparequant',
           'fmttable', 'printtable', 'showtable']

import io
import math
import numpy as np
import quantities as pq

import subprocess
import os

# Imports for displaying images in Jupyter.
# We can't use MathJax, since it only supports math environments
inlinetables = True
try:
  import wand.image
  import IPython.display
except ImportError:
    inlinetables = False
    import warnings
    warnings.warn("The wand module could not be imported."
                  " You won't see tables inlined in your notebook.")

def _isnum(val):
    return type(val) == int \
               or type(val) == float \
               or type(val) == complex \
               or hasattr(val, 'shape') and val.shape == ()

def _isquant(val):
    return isinstance(val, pq.Quantity) \
               or isinstance(val, pq.UncertainQuantity)

def _isweird(val):
    return val is None or val == 0.0 \
               or np.isnan(val) or np.isinf(val)


logten = math.log(10.0)
def fmtuncert(value, uncert=None,
              decimals=None, power=None, significance=None,
              pm=' +- ', ten=' 10**', tex=False, paren=False,
              nanempty=False):
    """Format integer / floating point values with uncertainty.
      Does not work with numpy arrays (use `numpy.vectorize`).
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
              before formatting (but after estimating power / decimals, so it
              is only useful with fixed decimals and power).

          pm: The separator between the value and its uncertainty.
          ten: The separator between the uncertainty (in parentheses) and
              the power.
          tex: If tex is True the power is surrounded by curly brackets.
          paren: If True, surround text with parentheses when uncertainty
              is shown, even if power is zero.
          nanempty: If True, an empty string is returned when value or
              uncert is numpy.nan

      Returns:
          A formatted string.

      Raises:
          ValueError: if `value` or `uncert` is not an int, float or complex.
    """
    if not _isnum(value):
        raise ValueError("`value` has to be an int, float or complex,"
                         " got {}".format(type(value)))
    if uncert is not None and not _isnum(uncert):
        raise ValueError("`uncert` has to be an int, float or complex,"
                         " got {}".format(type(uncert)))
    if nanempty and (np.isnan(value) or
                     uncert is not None and np.isnan(uncert)):
        return ""

    magn = abs(min(value.real, value.imag)) \
               if type(value) == complex \
               else abs(value)
    # estimate power
    if power is None:
        if _isweird(magn):
            power = 0
        else:
            power = int(math.floor(math.log(magn)/logten))
            if -2 <= power <= 3:
                power = 0
    value = value / 10**power
    magn = magn / 10**power
    uncert = uncert / 10**power if uncert is not None else None
    # estimate decimals
    if decimals is None:
        if _isweird(uncert): # four significant digits
            if _isweird(value): decimals = 1
            else:
                decimals = -int(math.floor(math.log(magn)/logten))
                decimals = max(0, decimals+3)
        else:
            umag = abs(min(uncert.real, uncert.imag)) \
                       if type(uncert) is complex \
                       else abs(uncert)
            decimals = -int(math.floor(math.log(umag)/logten))
            decimals = max(0, decimals+1)
            # fix rounding errors
            if round(uncert * 10**decimals) >= 100.0:
                decimals -= 1
    decimals = max(0, decimals)
    # round value and uncert to significance
    if significance is not None:
        s = 10**(significance - power)
        if not _isweird(value):
            value = round(value / s) * s
        if not _isweird(uncert):
            uncert = round(uncert / s) * s

    # fmt
    if uncert is None:
        num = "{0:.{1}f}".format(value, decimals)
    else:
        if type(uncert) == complex:
            num = "{0:.{2}f}{3}({1:.{2}f})".format(value, uncert, decimals, pm)
        else:
            num = "{0:.{2}f}{3}{1:.{2}f}".format(value, uncert, decimals, pm)
    if uncert is not None and (paren or power != 0):
        num = "("+num+")"
    if power != 0:
        if tex: num += ten + "{" + "%d"%power + "}"
        else: num += ten + "%d"%power
    return num


def fmtquant(quant, *args, **kwargs):
    """Give python Quantity object (or UncertainQuantity) and any
      arguments that fmtuncert takes.

      Additional arguments:
        units: Set True to append units (will make parentheses around
            values).  Formats as tex formula if `tex=True` is set.
            Defaults to True.

      Returns:
        formatted string

      Raises:
        ValueError: If quant or uncertainty are not a Quantity or
            have a built-in python number type.
    """
    tex = kwargs.get('tex', False)
    units = kwargs.pop('units', True)
    # value
    quant = 1.0 * quant
    if not _isquant(quant) and not _isnum(quant):
        raise ValueError("Value is not a Quantity or built-in number,"
                         " got type {}".format(type(quant)))
    value = quant.magnitude.item() if _isquant(quant) else quant
    # uncert
    uncert = None
    if len(args) > 0:
        uncert = args[0]
        args = args[1:]
    elif 'uncert' in kwargs:
        uncert = kwargs.pop('uncert')
    if uncert is not None:
        if not _isquant(uncert) and not _isnum(uncert):
            raise ValueError("Uncertainty is not a Quantity or built-in number,"
                             " got type {}".format(type(uncert)))
        uncert = 1.0 * uncert
        if _isquant(uncert):
            if _isquant(quant):
                uncert = uncert.rescale(quant.units)
            uncert = uncert.magnitude.item()
    elif hasattr(quant, 'uncertainty'):
        uncert = 1 * quant.uncertainty
        print(type(uncert))
    else:
        uncert = None
    # fmt
    assert len(args) < 9, "`paren` can only be keyword argument."
    paren = kwargs.pop('paren', False) or units
    s = fmtuncert(value, uncert, *args, paren=paren, **kwargs)
    if units and _isquant(quant) and quant.dimensionality != pq.dimensionless:
        if tex: s = s + ' ' + quant.dimensionality.latex[1:-1]
        else: s = s + ' ' + repr(quant.units)[13:]
    return s

# TODO
def fmtquant_vec(quants, uncerts, **kwargs):
    if not isinstance(quants, np.ndarray):
        raise ValueError("Expected array as `quants`")
    if not _isquant(quants) or uncerts is not None and not _isquant(uncerts):
        raise ValueError("Expected Quantities as input")
    if not isinstance(quants, np.ndarray):
        uncerts = np.array([uncerts] * len(quants))
    return list(fmtquant(q, u) for q,u in zip(quants, uncerts))

def comparequant(values1, values2, uncerts1=None, uncerts2=None):
    if uncerts1 is None:
        uncerts1 = [None]*len(values1)
    if uncerts2 is None:
        uncerts2 = [None]*len(values2)
    deviations = []
    rows = [] # formatted strings [(val1, val2, dev)]
    for v1, v2, u1, u2 in zip(values1, values2, uncerts1, uncerts2):
        v2.rescale(v1.units)
        if u1 is None: u1 = 1*v1.units
        if u2 is None: u2 = 1*v2.units
        dev = (v2 - v1) / np.sqrt(u1**2 + u2**2)
        deviations.append(dev)
        rows.append( (fmtquant(v1, u1), fmtquant(v2, u2), fmtquant(dev)) )
    # maximum fmtd string lengths per column
    col1 = max(len(s1) for s1, s2, s3 in rows)
    col2 = max(len(s2) for s1, s2, s3 in rows)
    col3 = max(len(s3) for s1, s2, s3 in rows)
    for s1, s2, s3 in rows:
        print(s1.rjust(col1)+' | '+s2.rjust(col2)+' | '+s3.rjust(col3))
    return deviations


################################################################################
## Format Tables

def _fmt_obj_column(header, values, fun=None):
    if not fun: fun = str
    col = list(fun(dat) for dat in values)
    return col

def _fmt_number_column(info, nanempty,
                       heading, values, uncert, decimals,
                       power=0, significance=None):
    # remove units
    if _isquant(values):
        if uncert is not None and _isquant(uncert):
            if uncert.units != values.units:
                if info:
                    print("INFO: rescaling uncertainty for", heading)
                uncert = uncert.rescale(values.units)
            uncert = uncert.magnitude
        values = values.magnitude
    # broadcast uncertainty
    if isinstance(uncert, np.ndarray):
        if uncert.size == 1:
            if info:
                print("INFO: broadcasting uncertainty for", heading)
            uncert = np.array([np.asscalar(uncert)] * len(values))
        if uncert.size != len(values):
            raise ValueError("Dimension mismatch for %s"%heading)
    elif not isinstance(uncert, list):
        if uncert is not None:
            if info:
                print("INFO: broadcasting uncertainty for", heading)
        uncert = np.array([uncert] * len(values))
    elif len(uncert) != len(values):
        raise ValueError("Dimension mismatch for %s"%heading)
    # format
    col = []
    for v, u in zip(values, uncert):
        f = fmtuncert(v, u, decimals,
                      power, significance,
                      pm=r" \pm ", ten='\cdot 10',
                      tex=True, nanempty=nanempty)
        col.append('$' + f + '$')
    return col

def fmttable(columns, caption="", tableno=1,
             columnformat=None, index=[],
             nanempty=True, info=True):
    """Format data as tex threeparttable.
      Table will have the same length as the longest column.

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
            as the longest column.  Defaults to an enumeration (when set to
            empty list).  No index is printed when `index is None`.
        nanempty: Passed on to fmtuncert(), defaults to True.
        info: Print to stdout when uncertainties are broadcasted,
            defaults to True.

      Returns: a string.

      Raises:
        ValueError: If types are not allowed or lengths of values and
            uncertainties don't match for a column.
    """
    coln = len(columns) # number of cols excluding index
    colN = coln+1 if index is not None else coln # and including index
    rown = max(len(col[1]) for col in columns)
    # create enumerating index or check given one
    if index == []: index = range(1, rown+1)
    if  index is not None and len(index) != rown:
        raise ValueError("Index must have length %d,"
                         " got %d"%(rown, len(index)))
    # create right aligned column format or check given one
    if not columnformat:
        columnformat = 'r' * (colN)
    if len(columnformat) != colN:
        raise ValueError("`columnformat` must have length %d,"
                         " got %d"%(colN, len(columnformat)))

    # format cells to strings
    fmtcols = []
    for coli, data in enumerate(columns):
        heading = data[0]
        if 2 <= len(data) <= 3:
            col = _fmt_obj_column(*data)
        elif 4 <= len(data) <= 6:
            col = _fmt_number_column(info, nanempty, *data)
        else:
            raise ValueError("Bad tuple for column %d"%(coli+1))
        if len(data) < rown:
            col.extend([""]*(rown-len(data)))
        fmtcols.append(col)

    # build string
    NL = '\n'
    s = io.StringIO()
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
    if index is not None:
        s.write("{} & ")
    s.write(" & ".join(headings) + r" \\" + NL)
    # data
    for rowi in range(rown):
        if index is not None:
            s.write(repr(index[rowi]) + " & ")
        s.write(" & ".join(fmtcols[coli][rowi] for coli in range(coln)))
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

def printtable(columns, caption="", tableno=1, name=None,
               columnformat=None, index=[],
               margins=[10, 10, 10, 10], keepcropped=False):
    """Shorthand for formatting and printing table.  See documentation of
      `fmttable()`, `printtex()` and `showtable()`.

      `name` defaults to 'tableTABLENO.pdf'.
    """
    tab = fmttable(columns, caption, tableno, columnformat, index)
    if name is None: name = "table{}".format(tableno)
    if not printtex(name, tab):
        return name
    showtable(name, margins, keepcropped)
    return name

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
\begin{document}
\pagestyle{empty}"""+NL)
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
        os.unlink(name+'.log')
        os.unlink(name+'.tex')
        os.unlink(name+'.aux')
        return True

def showtable(name, margins=[10, 10, 10, 10],
              keepcropped=False):
    """Crop the pages of a PDF file, then load it using ImageMagick
      (python wand) and display it in IPython / Jupyter.

      Arguments:
        name: basename of file.  The function looks for a file named
            `basename.pdf` and creates a file named `basename.cropped.pdf`.
        margins: A list or tuple of four integers giving the margins around the
            text in points ('pt').
    """
    if not keepcropped and not inlinetables:
        return # Nothing to do
    pdffile = name + '.pdf'
    cropfile = name + '.cropped.pdf'
    margins = ' '.join(str(m) for m in margins)
    try:
        proc = subprocess.Popen(['pdfcrop', '--margins', margins,
                                 pdffile, cropfile])
        proc.communicate()
    except FileNotFoundError as e:
        print('ERROR: pdfcrop not found, cannot display table in notebook')
    res = proc.returncode
    if res != 0:
        print("Cropping failed, return code: %d"%res)
        return
    if inlinetables:
        with wand.image.Image(filename=cropfile) as img:
            IPython.display.display(img)
    if not keepcropped:
        os.unlink(cropfile)
