"""Helpers for the Distant Viewing Toolkit tutorials.

A cut-down copy of the funs.py that accompanies the data science course notes
at https://taylor-arnold.github.io/fds2/book, keeping what the tutorials use:
the polars and plotnine conveniences and a few helpers for images.
"""

import inspect

import numpy as np
import plotnine as p9
import polars as pl

p9.theme_set(p9.theme_minimal())


# ============================================================
# Polars display utility
# ============================================================

def _patched_print(self, n=8):
    """Display a DataFrame showing a specific number of rows, as a terminal method in a chain."""
    with pl.Config(tbl_rows=n):
        try:
            # If running in a notebook, display the rich HTML table
            from IPython.display import display
            display(self)
        except ImportError:
            # Fallback for standard terminal environments
            print(self)

    # Returning None prevents Jupyter from double-printing
    # the dataframe if this is the last line in a cell.
    return None

pl.DataFrame.print = _patched_print


# ============================================================
# Plotnine x Polars integration
# ============================================================
# Lets ggplot() accept a Polars DataFrame directly, adds a .ggplot()
# method to DataFrame so a chain can open a plot without .pipe(), applies a
# default categorical ordering to string columns, supports `c.COLNAME`
# expressions inside aes(), and attaches every geom_/stat_/scale_/...
# component as a ggplot method so plots can be built as a single method chain.

_orig_ggplot_init = p9.ggplot.__init__

def _prepare_plot_data(df):
    """Apply categorical ordering to strings, in the order the values appear."""
    if not isinstance(df, pl.DataFrame):
        return df

    # Categorical logic
    string_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.String]
    if string_cols:
        exprs = []
        for col in string_cols:
            unique_vals = df[col].unique(maintain_order=True).drop_nulls().to_list()
            exprs.append(pl.col(col).cast(pl.Enum(unique_vals)))
        if exprs:
            df = df.with_columns(exprs)

    return df

def _patched_ggplot_init(self, *args, **kwargs):
    new_args = list(args)
    if len(args) > 0 and isinstance(args[0], pl.DataFrame):
        new_args[0] = _prepare_plot_data(args[0])
    elif 'data' in kwargs:
        kwargs['data'] = _prepare_plot_data(kwargs['data'])
    _orig_ggplot_init(self, *new_args, **kwargs)

p9.ggplot.__init__ = _patched_ggplot_init

def _df_ggplot(self, *args, **kwargs):
    """Start a plot from a DataFrame: df.ggplot(aes(...)) == ggplot(df, aes(...))."""
    return p9.ggplot(self, *args, **kwargs)

pl.DataFrame.ggplot = _df_ggplot

def _extract_names(val):
    """Recursively pull the string column name out of `c.COLNAME`-style Polars expressions."""
    if isinstance(val, pl.Expr):
        return val.meta.output_name()
    elif isinstance(val, (list, tuple)):
        # Handle lists of expressions like [c.id, c.date]
        return [_extract_names(v) for v in val]
    elif isinstance(val, dict):
        return {k: _extract_names(v) for k, v in val.items()}
    return val

_orig_aes_init = p9.aes.__init__

def _patched_aes_init(self, *args, **kwargs):
    """Wrap aes() so it accepts `c.COLNAME` expressions in place of column-name strings."""
    new_args = tuple(_extract_names(arg) for arg in args)
    new_kwargs = {k: _extract_names(v) for k, v in kwargs.items()}
    _orig_aes_init(self, *new_args, **new_kwargs)

p9.aes.__init__ = _patched_aes_init

# Dynamically attach every geom_/stat_/theme/scale_/facet_/coord_/guides/labs/
# annotate component to the ggplot class as a method, so that
# ggplot(...).geom_point() runs ggplot(...) + geom_point() behind the scenes.
_component_prefixes = ('geom_', 'stat_', 'theme', 'scale_', 'facet_', 'coord_', 'guides', 'labs', 'annotate')

for _name in dir(p9):
    _obj = getattr(p9, _name)
    if inspect.isclass(_obj) and _name.startswith(_component_prefixes):

        def _make_method(component_cls):
            def method(self, *args, **kwargs):
                return self + component_cls(*args, **kwargs)
            return method

        setattr(p9.ggplot, _name, _make_method(_obj))

# theme() and guides() need more than a plain method. ggplot.__init__ stores a
# component of each of those names on the instance (self.theme = theme_get(),
# self.guides = guides()), and an instance attribute hides a class method of the
# same name, so the methods attached above are unreachable for these two. A data
# descriptor, however, takes precedence over the instance dictionary. Each one
# below hands plotnine back the component it stored, so `plot.theme` keeps
# working internally, and records the plot the component came from so that
# *calling* the component adds a fresh one to that plot. The owner is recorded as
# .plot, the name plotnine itself uses (theme.setup) and the one its __deepcopy__
# knows not to follow.

def _make_chainable_component(name):
    component_cls = getattr(p9, name)
    store = f"_{name}"

    class _ChainableComponent:
        def __get__(self, plot, owner=None):
            if plot is None:
                return self
            try:
                component = plot.__dict__[store]
            except KeyError:
                raise AttributeError(name) from None
            component.plot = plot
            return component

        def __set__(self, plot, component):
            plot.__dict__[store] = component

    def _call(self, *args, **kwargs):
        plot = getattr(self, "plot", None)
        if plot is None:
            raise TypeError(f"{name}() is only callable as a method of a ggplot object")
        return plot + component_cls(*args, **kwargs)

    component_cls.__call__ = _call
    setattr(p9.ggplot, name, _ChainableComponent())

for _name in ('theme', 'guides'):
    _make_chainable_component(_name)


# ============================================================
# Reshaping
# ============================================================
# Lets .pivot() and .unpivot() accept `c.COLNAME` expressions in place of
# column-name strings, reusing the _extract_names() helper defined above.

_orig_pivot = pl.DataFrame.pivot
_orig_unpivot = pl.DataFrame.unpivot

def _patched_pivot(self, *args, **kwargs):
    """Wrap .pivot() so it accepts `c.COLNAME` expressions in place of column-name strings."""
    new_args = tuple(_extract_names(arg) for arg in args)
    new_kwargs = {k: _extract_names(v) for k, v in kwargs.items()}
    return _orig_pivot(self, *new_args, **new_kwargs)

def _patched_unpivot(self, *args, **kwargs):
    """Wrap .unpivot() so it accepts `c.COLNAME` expressions in place of column-name strings."""
    new_args = tuple(_extract_names(arg) for arg in args)
    new_kwargs = {k: _extract_names(v) for k, v in kwargs.items()}
    return _orig_unpivot(self, *new_args, **new_kwargs)

pl.DataFrame.pivot = _patched_pivot
pl.DataFrame.unpivot = _patched_unpivot



# ============================================================
# Images
# ============================================================

def dot_product(a, b):
    """Row-wise dot product of two list-typed Polars columns."""
    return (a * b).list.sum()


class DSImage:
    """Static helpers for summarizing and displaying image datasets."""

    def __new__(cls, *args, **kwargs):
        raise TypeError("DSImage is a static utility class and cannot be instantiated")

    @staticmethod
    def read(path):
        """Load an image file as an array of pixels: rows, columns, then red, green and blue."""
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def show(img):
        """Display an array of red, green and blue pixels as an image in the notebook."""
        from PIL import Image
        try:
            from IPython.display import display
            display(Image.fromarray(img))
        except ImportError:
            Image.fromarray(img).show()

    @staticmethod
    def compute_colors(img_hsv):
        """Return the percentage of pixels in an HSV image falling into each named hue bucket."""
        import cv2
        h, s, v = cv2.split(img_hsv)
        valid = (s >= 50) & (v >= 50)

        color_ranges = {
            'red': ((h <= 10) | (h >= 170)) & valid,
            'orange': (h > 10) & (h <= 25) & valid,
            'yellow': (h > 25) & (h <= 35) & valid,
            'green': (h > 35) & (h <= 85) & valid,
            'cyan': (h > 85) & (h <= 100) & valid,
            'blue': (h > 100) & (h <= 130) & valid,
            'purple': (h > 130) & (h <= 145) & valid,
            'magenta': (h > 145) & (h < 170) & valid,
        }

        total_pixels = img_hsv.shape[0] * img_hsv.shape[1]

        percentages = {color: np.sum(mask) / total_pixels * 100
                       for color, mask in color_ranges.items()}

        percentages['neutral'] = np.sum(~valid) / total_pixels * 100
        return percentages

    @staticmethod
    def plot_image_grid(df, ncol=10, label_name="label", filepath="filepath", limit=100):
        """Display up to `limit` images from `df` in a grid, labeled from `label_name` if present."""
        import matplotlib.pyplot as plt
        from PIL import Image
        import math

        df = df.head(limit)
        n = df.height
        if n == 0:
            return

        nrow = math.ceil(n / ncol)

        paths = df.select(filepath).to_series().to_list()
        labels = None
        if label_name is not None and label_name in df.columns:
            labels = df.select(label_name).to_series().to_list()

        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        axes = np.array(axes).ravel()

        for i, ax in enumerate(axes):
            if i < n:
                img = Image.open(paths[i])
                cmap = "gray" if img.mode == "L" else None
                ax.imshow(img, cmap=cmap)

                if labels is not None:
                    ax.set_title(str(labels[i]), fontsize=8)

                w, h = img.size
                max_dim = max(h, w)
                ax.set_xlim(-0.5 + (w - max_dim) / 2, w - 0.5 + (max_dim - w) / 2)
                ax.set_ylim(h - 0.5 + (max_dim - h) / 2, -0.5 + (h - max_dim) / 2)
                ax.set_aspect("equal")
                ax.axis("off")
            else:
                ax.axis("off")

        fig.set_constrained_layout(True)
        plt.show()
