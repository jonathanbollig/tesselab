"""
SVD and MCR-ALS for any matrix: the input is matched by shape (n × m), so the same nodes work on UV spectra over
irradiation steps, transient absorption over delays, or a plain table.

SVD: how many independent components the data hold (singular values), what they look like (components along m) and
how their weights change (along n). MCR-ALS (pymcr): non-negative concentrations C (n × species) and spectra S
(species × m) with C·S ≈ the matrix; it runs only on its Run button, from random non-negative starting spectra
(seed: reproducible) or from given ones (input "initial").
"""
import numpy as np
import xarray as xr

import tesselab
from tesselab import P, Plot, PlotSpec, Axis, Line, Lines, Points, Heatmap, Signature, Var, Result

MATRIX = Signature(matrix=Var("n", "m"), row_pos=Var("n", optional=True), col_pos=Var("m", optional=True))


def _axis_info(data, name, dim):
    """(values, label, unit) of the positions along a dimension: the bound variable if there is one (with its
    original name and unit), else the index."""
    names = data.attrs.get("tesselab_names", {})
    if name in data.variables and name in names:
        return data[name].values, names[name].replace("_", " "), data[name].attrs.get("units", "")
    original = data.attrs.get("tesselab_dims", {}).get(dim, dim)
    return np.arange(data.sizes[dim], dtype=float), f"{original} (index)", ""


def _clean(data):
    """The matrix without rows and columns that contain NaN (e.g. wavelengths outside a buffer's range)."""
    matrix = data.matrix.values.astype(float)
    rows = ~np.isnan(matrix).all(axis=1)
    columns = ~np.isnan(matrix[rows]).any(axis=0)
    return matrix, rows, columns


class SVD(tesselab.Node):
    name = "SVD"
    views = ["svd-view"]
    help = "Singular value decomposition: how many independent components the data hold, and what they look like."
    inputs = {"data": MATRIX}
    params = [P.Int("components", 3, min=1, max=20, help="Components shown and used for the reconstruction."),
              P.Bool("subtract_mean", False, help="Subtract the mean along n first (then the first component is "
                                                  "the largest change, not the average).")]
    outputs = ["svd"]
    key_results = [Result("residual", label="residual", unit="%", digits=3)]

    def run(self, inputs, params):
        data = inputs.data
        matrix, rows, columns = _clean(data)
        used = matrix[np.ix_(rows, columns)]
        if used.size == 0:
            raise ValueError("The matrix is empty after removing rows and columns with missing values.")
        if params.subtract_mean:
            used = used - used.mean(axis=0)
        u, s, vt = np.linalg.svd(used, full_matrices=False)
        k = min(params.components, len(s))
        reconstruction = (u[:, :k] * s[:k]) @ vt[:k]
        residual = np.full(matrix.shape, np.nan)
        residual[np.ix_(rows, columns)] = used - reconstruction
        weights = np.full((matrix.shape[0], k), np.nan)
        weights[rows] = u[:, :k] * s[:k]
        components = np.full((k, matrix.shape[1]), np.nan)
        components[:, columns] = vt[:k]
        norm = np.linalg.norm(used)
        out = xr.Dataset({
            "singular_values": ("index", s),
            "weights": (("n", "component"), weights),
            "components": (("component", "m"), components),
            "residual_matrix": (("n", "m"), residual),
            "residual": ((), float(np.linalg.norm(used - reconstruction) / norm * 100) if norm else np.nan),
        }, coords={"index": np.arange(1, len(s) + 1), "component": np.arange(1, k + 1)})
        for name, dim in (("row_pos", "n"), ("col_pos", "m")):
            values, label, unit = _axis_info(data, name, dim)
            out.coords[name] = (dim, values, {"units": unit, "label": label})
        return {"svd": out}


class MCR(tesselab.Node):
    views = ["mcr-view"]
    name = "MCR-ALS"
    help = "Multivariate curve resolution: non-negative concentrations and spectra whose product fits the data."
    inputs = {"data": MATRIX,
              "initial": Signature(spectra=Var("species", "m"), optional=True)}
    params = [P.Int("species", 3, min=1, max=10, help="Number of components."),
              P.Int("max_iter", 200, min=1, help="Iterations of each run."),
              P.Int("runs", 5, min=1, help="Runs from different random starts; the best fit is kept."),
              P.Seed("seed", 0, help="Random starts; the same seed gives the same result.")]
    outputs = ["mcr"]
    key_results = [Result("lof", label="lack of fit", unit="%", digits=3)]
    run_on = "button"

    def run(self, inputs, params):
        try:
            from pymcr.mcr import McrAR
            from pymcr.constraints import ConstraintNonneg
        except ImportError:
            raise RuntimeError("MCR-ALS needs the package pymcr (pip install pymcr).") from None
        import logging
        logging.getLogger("pymcr.mcr").disabled = True
        data = inputs.data
        matrix, rows, columns = _clean(data)
        used = matrix[np.ix_(rows, columns)]
        rng = np.random.default_rng(params.seed)
        starts = []
        if "initial" in inputs:
            initial = inputs.initial.spectra.values
            if initial.shape[1] != matrix.shape[1]:
                raise ValueError(f"The initial spectra have {initial.shape[1]} points along m, the data "
                                 f"{matrix.shape[1]}: they must be on the same axis.")
            starts.append(initial[:, columns])
        while len(starts) < params.runs:
            starts.append(rng.random((params.species, used.shape[1])) * np.nanmax(np.abs(used)))
        best = None
        for start in starts:
            mcr = McrAR(max_iter=params.max_iter, c_constraints=[ConstraintNonneg()],
                        st_constraints=[ConstraintNonneg()], tol_increase=10)
            mcr.fit(used, ST=start)
            fit = mcr.C_opt_ @ mcr.ST_opt_
            lof = float(np.sqrt(np.sum((used - fit) ** 2) / np.sum(used ** 2)) * 100)
            if best is None or lof < best[0]:
                best = (lof, mcr.C_opt_, mcr.ST_opt_, used - fit)
        lof, c, st, residual = best
        species = st.shape[0]
        full_c = np.full((matrix.shape[0], species), np.nan)
        full_c[rows] = c
        full_st = np.full((species, matrix.shape[1]), np.nan)
        full_st[:, columns] = st
        full_residual = np.full(matrix.shape, np.nan)
        full_residual[np.ix_(rows, columns)] = residual
        out = xr.Dataset({"concentrations": (("n", "species"), full_c), "spectra": (("species", "m"), full_st),
                          "residual_matrix": (("n", "m"), full_residual), "lof": ((), lof)},
                         coords={"species": np.arange(1, species + 1)})
        for name, dim in (("row_pos", "n"), ("col_pos", "m")):
            values, label, unit = _axis_info(data, name, dim)
            out.coords[name] = (dim, values, {"units": unit, "label": label})
        return {"mcr": out}


# --- views ------------------------------------------------------------------------------------------------------------
class SVDView(tesselab.View):
    name = "SVD view"
    help = "The singular values, the components and their weights, and what the components miss."
    inputs = {"svd": Signature(singular_values=Var("index"), components=Var("component", "m"),
                               weights=Var("n", "component"), residual_matrix=Var("n", "m"),
                               help="The SVD of a matrix.")}
    plots = [
        Plot("values", "Singular values", draw=lambda ctx: PlotSpec(
            x=Axis("Index"), y=Axis("Singular value", scale="log"),
            marks=[Points(ctx.inputs.svd.index, ctx.inputs.svd.singular_values, name="singular values")],
            zero_line=False), help="Large values: components that carry signal; the rest is noise."),
        Plot("components", "Components", draw=lambda ctx: PlotSpec(
            x=Axis(ctx.inputs.svd.col_pos.attrs["label"], ctx.inputs.svd.col_pos.attrs["units"],
                   display="nm" if ctx.inputs.svd.col_pos.attrs["units"] == "m" else None),
            y=Axis("Component"),
            marks=[Lines(ctx.inputs.svd.col_pos, ctx.inputs.svd.components,
                         labels=[f"{c}" for c in ctx.inputs.svd.component.values])],
            legend="top right"), help="The first components (rows of Vᵀ) along m."),
        Plot("weights", "Weights", draw=lambda ctx: PlotSpec(
            x=Axis(ctx.inputs.svd.row_pos.attrs["label"], ctx.inputs.svd.row_pos.attrs["units"]),
            y=Axis("Weight"),
            marks=[Lines(ctx.inputs.svd.row_pos, ctx.inputs.svd.weights.transpose("component", "n"),
                         labels=[f"{c}" for c in ctx.inputs.svd.component.values])],
            legend="top right"), help="How strongly each component is present along n (U·S)."),
        Plot("residual", "Residual of the reconstruction", draw=lambda ctx: PlotSpec(
            x=Axis(ctx.inputs.svd.row_pos.attrs["label"], ctx.inputs.svd.row_pos.attrs["units"]),
            y=Axis(ctx.inputs.svd.col_pos.attrs["label"], ctx.inputs.svd.col_pos.attrs["units"],
                   display="nm" if ctx.inputs.svd.col_pos.attrs["units"] == "m" else None),
            marks=[Heatmap(ctx.inputs.svd.row_pos, ctx.inputs.svd.col_pos, ctx.inputs.svd.residual_matrix,
                           colorbar="residual")], zero_line=False),
            help="The data minus their reconstruction from the components shown: structure here is signal the "
                 "components miss."),
    ]


class MCRView(tesselab.View):
    name = "MCR-ALS view"
    help = "The resolved spectra and concentrations, and the residual."
    inputs = {"mcr": Signature(spectra=Var("species", "m"), concentrations=Var("n", "species"),
                               residual_matrix=Var("n", "m"), help="The MCR-ALS result.")}
    plots = [
        Plot("spectra", "MCR spectra", draw=lambda ctx: PlotSpec(
            x=Axis(ctx.inputs.mcr.col_pos.attrs["label"], ctx.inputs.mcr.col_pos.attrs["units"],
                   display="nm" if ctx.inputs.mcr.col_pos.attrs["units"] == "m" else None),
            y=Axis("Spectrum"),
            marks=[Lines(ctx.inputs.mcr.col_pos, ctx.inputs.mcr.spectra, labels=[f"{s}" for s in ctx.inputs.mcr.species.values])],
            legend="top right"), help="The resolved spectra (rows of S); their scale is not determined by the data."),
        Plot("concentrations", "MCR concentrations", draw=lambda ctx: PlotSpec(
            x=Axis(ctx.inputs.mcr.row_pos.attrs["label"], ctx.inputs.mcr.row_pos.attrs["units"]),
            y=Axis("Concentration"),
            marks=[Lines(ctx.inputs.mcr.row_pos, ctx.inputs.mcr.concentrations.transpose("species", "n"),
                         labels=[f"{s}" for s in ctx.inputs.mcr.species.values])], legend="top right"),
            help="How much of each species is present along n."),
        Plot("residual", "MCR residual", draw=lambda ctx: PlotSpec(
            x=Axis(ctx.inputs.mcr.row_pos.attrs["label"], ctx.inputs.mcr.row_pos.attrs["units"]),
            y=Axis(ctx.inputs.mcr.col_pos.attrs["label"], ctx.inputs.mcr.col_pos.attrs["units"],
                   display="nm" if ctx.inputs.mcr.col_pos.attrs["units"] == "m" else None),
            marks=[Heatmap(ctx.inputs.mcr.row_pos, ctx.inputs.mcr.col_pos, ctx.inputs.mcr.residual_matrix, colorbar="residual")],
            zero_line=False), help="The data minus C·S."),
    ]
