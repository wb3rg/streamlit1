#
# MIT License
#
# Copyright (c) 2015 Jonathan Shore
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import numpy as np
import pandas as pd
from plotnine import *
from plotnine.scales import scale_x_datetime
from plotnine.coords import coord_cartesian
from plotnine.facets import facet_grid
from plotnine.facets.facet_grid import parse_grid_facets_old as parse_grid_facets
import plotnine
from plotnine.exceptions import PlotnineError
from .ggplot_internals import facet, layout_null, combine_vars, add_missing_facets
from .ggplot_internals import eval_facet_vars

from .ggplot_internals import facet, layout_null

def ninteraction(df, drop=False):
    """
    Compute a unique numeric id for each unique row in a data frame.
    """
    if len(df.columns) == 0:
        return np.zeros(len(df))
    if drop:
        df = df.drop_duplicates()
    return pd.factorize(df.apply(tuple, axis=1))[0]

def add_margins(data, vars, margins=True):
    """
    Add margins to a data frame.
    """
    if not margins or not vars:
        return data
    
    all_vars = []
    for v in vars:
        if isinstance(v, list):
            all_vars.extend(v)
        else:
            all_vars.append(v)
    
    margin_vars = [v for v in all_vars if v in data.columns]
    if not margin_vars:
        return data
    
    # Add a copy of the data with each variable set to None
    margin_dfs = [data]
    for v in margin_vars:
        df_copy = data.copy()
        df_copy[v] = None
        margin_dfs.append(df_copy)
    
    return pd.concat(margin_dfs, ignore_index=True)

def cross_join(df1, df2):
    """
    Compute a cross join between two data frames.
    """
    df1['_tmpkey'] = 1
    df2['_tmpkey'] = 1
    merged = pd.merge(df1, df2, on='_tmpkey')
    merged = merged.drop('_tmpkey', axis=1)
    return merged

def match(x, table, start=0):
    """
    Return a vector of the positions of (first) matches of x in table.
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(table, pd.Series):
        table = pd.Series(table)
    
    result = pd.Series(np.nan, index=x.index)
    for i, val in enumerate(table, start=start):
        mask = (x == val) & result.isna()
        result[mask] = i
    return result.astype(int)

def join_keys(x, y, by):
    """
    Join two data frames by common keys.
    """
    if not by:
        return {'x': pd.Series(), 'y': pd.Series()}
    
    x_vals = x[by].apply(tuple, axis=1)
    y_vals = y[by].apply(tuple, axis=1)
    
    return {'x': x_vals, 'y': y_vals}

def scale_x_datetime_auto(name='', breaks=None, labels=None, limits=None,
                         expand=None):
    """
    Create a datetime scale with automatic breaks and labels.
    """
    return scale_x_datetime(name=name, breaks=breaks, labels=labels,
                          limits=limits, expand=expand)

def new_grid(data, x, y, scales='free_y', space='free_y'):
    """
    Create a new grid facet.
    """
    return facet_grid(f'{y} ~ {x}', scales=scales, space=space)


class new_grid(facet):
    """
    Wrap 1D Panels onto 2D surface

    Parameters
    ----------
    facets : str | tuple | list
        A formula with the rows (of the tabular display) on
        the LHS and the columns (of the tabular display) on
        the RHS; the dot in the formula is used to indicate
        there should be no faceting on this dimension
        (either row or column). If a tuple/list is used, it
        must of size two, the elements of which must be
        strings or lists. If string formula is not processed
        as you may expect, use tuple/list. For example, the
        follow two specifications are equivalent::

            'func(var4) ~ func(var1+var3) + func(var2)'
            ['func(var4)', ('func(var1+var3)', 'func(var2)')]

        There may be cases where you cannot use a
        use a pure string formula, e.g.::

            ['var4', ('var1+var3', 'var2')]

    scales : str in ``['fixed', 'free', 'free_x', 'free_y']``
        Whether ``x`` or ``y`` scales should be allowed (free)
        to vary according to the data along rows or columns.
        Default is ``'fixed'``, the same scales for all the
        panels.
    space : str in ``['fixed', 'free', 'free_x', 'free_y']``
        Whether the ``x`` or ``y`` sides of the panels
        should have the size. It also depends to the
        ``scales`` parameter. Default is ``'fixed'``.
        This setting is not yet supported.
    shrink : bool
        Whether to shrink the scales to the output of the
        statistics instead of the raw data. Default is ``True``.
    labeller : str | function
        How to label the facets. If it is a ``str``, it should
        be one of ``'label_value'`` ``'label_both'`` or
        ``'label_context'``. Default is ``'label_value'``
    as_table : bool
        If ``True``, the facets are laid out like a table with
        the highest values at the bottom-right. If ``False``
        the facets are laid out like a plot with the highest
        value a the top-right. Default it ``True``.
    drop : bool
        If ``True``, all factor levels not used in the data
        will automatically be dropped. If ``False``, all
        factor levels will be shown, regardless of whether
        or not they appear in the data. Default is ``True``.
    height_ratios: for example [2, 1]
        list of heights (relative ratio) for each vertical row in facet
    width_ratios: for example [1, 1]
        list of widths (relative ratio) for each horizontal column in facet
    """

    def __init__(self, facets, margins=False, scales='fixed',
                 space='fixed', shrink=True, labeller='label_value',
                 as_table=True, drop=True, height_ratios=None, width_ratios=None):
        facet.__init__(
            self, scales=scales, shrink=shrink, labeller=labeller,
            as_table=as_table, drop=drop, height_ratios=height_ratios, width_ratios=width_ratios)
        self.rows, self.cols = parse_grid_facets(facets)
        self.margins = margins
        self.space_free = {'x': space in ('free_x', 'free'),
                           'y': space in ('free_y', 'free')}
        self.num_vars_x = len(self.cols)
        self.num_vars_y = len(self.rows)

    def compute_layout(self, data):
        if not self.rows and not self.cols:
            return layout_null()

        base_rows = combine_vars(data, self.plot.environment,
                                 self.rows, drop=self.drop)

        if not self.as_table:
            # Reverse the order of the rows
            base_rows = base_rows[::-1]
        base_cols = combine_vars(data, self.plot.environment,
                                 self.cols, drop=self.drop)

        base = cross_join(base_rows, base_cols)

        if self.margins:
            base = add_margins(base, [self.rows, self.cols], self.margins)
            base = base.drop_duplicates().reset_index(drop=True)

        n = len(base)
        panel = ninteraction(base, drop=True)
        panel = pd.Categorical(panel, categories=range(1, n+1))

        if self.rows:
            rows = ninteraction(base[self.rows], drop=True)
        else:
            rows = 1

        if self.cols:
            cols = ninteraction(base[self.cols], drop=True)
        else:
            cols = 1

        layout = pd.DataFrame({'PANEL': panel,
                               'ROW': rows,
                               'COL': cols})
        layout = pd.concat([layout, base], axis=1)
        layout = layout.sort_values('PANEL')
        layout.reset_index(drop=True, inplace=True)

        # Relax constraints, if necessary
        layout['SCALE_X'] = layout['COL'] if self.free['x'] else 1
        layout['SCALE_Y'] = layout['ROW'] if self.free['y'] else 1
        layout['AXIS_X'] = layout['ROW'] == layout['ROW'].max()
        layout['AXIS_Y'] = layout['COL'] == layout['COL'].min()

        self.nrow = layout['ROW'].max()
        self.ncol = layout['COL'].max()
        return layout

    def map(self, data, layout):
        if not len(data):
            data['PANEL'] = pd.Categorical(
                [],
                categories=layout['PANEL'].cat.categories,
                ordered=True)
            return data

        vars = [x for x in self.rows + self.cols]
        margin_vars = [list(data.columns & self.rows),
                       list(data.columns & self.cols)]
        data = add_margins(data, margin_vars, self.margins)

        facet_vals = eval_facet_vars(data, vars, self.plot.environment)
        data, facet_vals = add_missing_facets(data, layout,
                                              vars, facet_vals)

        # assign each point to a panel
        if len(facet_vals) == 0:
            # Special case of no facetting
            data['PANEL'] = 1
        else:
            keys = join_keys(facet_vals, layout, vars)
            data['PANEL'] = match(keys['x'], keys['y'], start=1)

        data = data.sort_values('PANEL', kind='mergesort')

        # matching dtype and
        # the categories(panel numbers) for the data should be in the
        # same order as the panels. i.e the panels are the reference,
        # they "know" the right order
        data['PANEL'] = pd.Categorical(
            data['PANEL'],
            categories=layout['PANEL'].cat.categories,
            ordered=True)

        data.reset_index(drop=True, inplace=True)
        return data

    def spaceout_and_resize_panels(self):
        """
        Adjust the spacing between the panels and resize them
        to meet the aspect ratio
        """
        ncol = self.ncol
        nrow = self.nrow
        figure = self.figure
        theme = self.theme
        get_property = theme.themeables.property

        left = figure.subplotpars.left
        right = figure.subplotpars.right
        top = figure.subplotpars.top
        bottom = figure.subplotpars.bottom
        wspace = figure.subplotpars.wspace
        W, H = figure.get_size_inches()

        try:
            spacing_x = get_property('panel_spacing_x')
        except KeyError:
            spacing_x = 0.1

        try:
            spacing_y = get_property('panel_spacing_y')
        except KeyError:
            spacing_y = 0.1

        try:
            aspect_ratio = get_property('aspect_ratio')
        except KeyError:
            # If the panels have different limits the coordinates
            # cannot compute a common aspect ratio
            if not self.free['x'] and not self.free['y']:
                aspect_ratio = self.coordinates.aspect(
                    self.layout.panel_params[0])
            else:
                aspect_ratio = None

        # The goal is to have equal spacing along the vertical
        # and the horizontal. We use the wspace and compute
        # the appropriate hspace. It would be a lot easier if
        # MPL had a better layout manager.

        # width of axes and height of axes
        w = ((right-left)*W - spacing_x*(ncol-1)) / ncol
        h = ((top-bottom)*H - spacing_y*(nrow-1)) / nrow

        # aspect ratio changes the size of the figure
        if aspect_ratio is not None:
            h = w*aspect_ratio
            H = (h*nrow + spacing_y*(nrow-1)) / (top-bottom)
            figure.set_figheight(H)

        # spacing
        wspace = spacing_x/w
        hspace = spacing_y/h
        figure.subplots_adjust(wspace=wspace, hspace=hspace)

    def draw_label(self, layout_info, ax):
        """
        Draw facet label onto the axes.

        This function will only draw labels if they are needed.

        Parameters
        ----------
        layout_info : dict-like
            Layout information. Row from the `layout` table.
        ax : axes
            Axes to label
        """
        toprow = layout_info['ROW'] == 1
        rightcol = layout_info['COL'] == self.ncol

        if toprow and len(self.cols):
            label_info = layout_info[list(self.cols)]
            label_info._meta = {'dimension': 'cols'}
            label_info = self.labeller(label_info)
            self.draw_strip_text(label_info, 'top', ax)

        if rightcol and len(self.rows):
            label_info = layout_info[list(self.rows)]
            label_info._meta = {'dimension': 'rows'}
            label_info = self.labeller(label_info)
            self.draw_strip_text(label_info, 'right', ax)


def parse_grid_facets(facets):
    """
    Parse faceting specification into row and column variables.
    
    Parameters
    ----------
    facets : str | tuple | list
        Faceting specification
        
    Returns
    -------
    rows, cols : tuple
        Row and column faceting variables
    """
    if isinstance(facets, (tuple, list)):
        rows, cols = facets
    else:
        try:
            rows, cols = facets.split('~')
        except (AttributeError, ValueError):
            rows, cols = '', facets

    rows = rows.strip()
    cols = cols.strip()

    if not rows:
        rows = []
    elif '+' in rows:
        rows = [var.strip() for var in rows.split('+')]
    else:
        rows = [rows]

    if not cols:
        cols = []
    elif '+' in cols:
        cols = [var.strip() for var in cols.split('+')]
    else:
        cols = [cols]

    return rows, cols


def ensure_var_or_dot(formula_term):
    """
    Ensure that a non specified formula term is transformed into a dot.
    """
    return formula_term if formula_term else '.'
