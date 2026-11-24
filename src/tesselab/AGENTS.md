# Writing a Tesselab module
This is the protocol for Tesselab modules, written for whoever builds one: a person or an LLM. It is complete: a module that follows it and passes `tesselab check` works in Tesselab. Protocol version: `api = 1`.
## In one paragraph
A **module** is a folder with a manifest (`module.toml`) and Python code. It supplies **nodes** of three kinds: **readers** read a file and return the data as measured; **compute nodes** compute ("subtract the background", "fit two lifetimes"); **views** show results. A reader or compute node declares its **inputs** (data, matched by shape), its **parameters** (typed; Tesselab builds the form) and its **outputs** (xarray Datasets), and computes in one function, `run(inputs, params)`. A view takes the outputs of any stage as its inputs and declares **plots** (descriptions; Tesselab draws them) and the **interactions** on them; each view is a section of the analysis's card, and only views have plots. A user combines nodes into an **analysis** (a graph, saved as a small JSON file); a module ships **templates**, ready-made analyses for its kind of data. Everything else (window, drawing, previews of any output, undo, saving, workers, cache, export) is Tesselab's.
## The loop
1. `tesselab new module <folder>` writes a minimal module that already passes the check. Start from it.
2. Put a **small real dataset** into `sample/` (one data folder or file, a few hundred kB at most).
3. Write `analysis.py` (the computation, plain numpy/xarray, no Tesselab, usable from a notebook) and `nodes.py` (the nodes that call it).
4. Run `tesselab check <folder>`. It loads the sample, runs every node twice, checks every output, renders every plot to `<folder>/check/*.png` (at scale 1 and 1.5) and writes an export. Fix what it reports.
5. **Look at the PNGs** and `check/export_*_data.json`: do they show what the user asked for? A passing check does not mean the plots are right.
6. The user opens the data in Tesselab (put the module folder into the user module folder, `~/.local/share/tesselab/modules/` on Linux, or start with `tesselab --modules <parent folder>`) and tells you what to change. The window follows your edits: when a file of the module changes, it reloads the module and runs its nodes again within a second (Modules → Follow changes on disk; without it, the user presses Modules → Reload modules). A syntax error shows on the module's nodes and goes away with the next good save.
## The folder
```
my_module/
    module.toml      the manifest
    nodes.py         the nodes: the only file that imports tesselab
    analysis.py      the computation (or an installed package)
    templates/       ready-made analyses, e.g. default.json
    sample/          a small real dataset; check runs on it
    tests/           optional: pytest tests with known answers (check runs them)
    README.md        what the module is for
```
The folder is imported as a package of its own, so files import each other relatively: `from . import analysis`.
## module.toml
```toml
[module]
id = "org.example.transient-absorption"  # never changes: analysis files refer to it. Letters, digits, . - _
name = "Transient absorption"
version = "0.1.0"
api = 1
description = "One sentence: what data, what result."
authors = ["..."]
requires = ["numpy", "xarray", "scipy"]  # Python packages (pip names); missing ones are reported, never installed
platforms = ["linux", "windows"]          # optional

[[nodes]]
id = "harpia"                    # full id: org.example.transient-absorption/harpia
class = "nodes.py:HarpiaReader"
files = ["*.dat"]                # optional: files this node reads

[[templates]]
id = "global-analysis"
name = "TA global analysis"
file = "templates/global_analysis.json"
help = "What it is for."
recognize = { all = ["*.dat"] }  # data folders it is offered for: all = every glob matches, any = one does
fill = "nodes.py:fill_template"  # optional: function(folder, project_root) -> placeholder values
detect = "nodes.py:detect"       # optional: function(folder) -> bool, a closer look (fast: under 50 ms)
```
Globs are relative to the data folder: `"powers"` (a subfolder), `"data/*.csv"`.
## A node
```python
import numpy as np
import xarray as xr
import tesselab
from tesselab import P, Plot, PlotSpec, Axis, Line, Points, Region, Signature, Var, Result, RangeSelect
from . import analysis

class Decay(tesselab.Node):                            # a compute node: no plots
    name = "Single exponential"                        # shown in the window
    help = "Fits A·exp(−t/τ) + c to a kinetic trace."  # hover text
    inputs = {"trace": Signature(signal=Var("t"), t=Var("t", units="s"), help="The kinetic to fit.")}
    params = [P.Range("fit_range", [0.0, 1e-9], unit="s", display="ps", help="The delays fitted."),
              P.Int("max_iterations", 200, min=10, advanced=True)]
    outputs = ["fit"]
    views = ["decay-view"]                             # added with it (Add node → Single exponential)
    key_results = [Result("tau", label="τ", unit="s", display="ps", error="tau_err", output="fit")]

    def run(self, inputs, params):
        t, y = inputs.trace.t.values, inputs.trace.signal.values
        tau, tau_err, model = analysis.fit_decay(t, y, params.fit_range, params.max_iterations)
        if tau_err > tau:
            tesselab.warning("τ is not determined by the data.", code="tau-undetermined")
        return {"fit": xr.Dataset({"signal": ("t", y), "model": ("t", model), "tau": tau, "tau_err": tau_err},
                                  coords={"t": ("t", t, {"units": "s"})})}

def draw_fit(ctx):
    fit = ctx.inputs.fit
    low, high = ctx.params.fit_range
    return PlotSpec(x=Axis("Delay", "s", display="ps"), y=Axis("ΔA", "OD", display="mOD"),
                    marks=[Region([low], [high]), Points(fit.t, fit.signal, name="data"),
                           Line(fit.t, fit.model, name="fit", style=tesselab.Style(color="fit"))])

class DecayView(tesselab.View):                        # a view: inputs and plots, no outputs, no run()
    name = "Decay"
    inputs = {"fit": Signature(signal=Var("t"), model=Var("t"), help="The fit.")}
    params = [P.Range("fit_range", [0.0, 1e-9], unit="s", display="ps",
                      help="Shown and dragged here; bound to the analysis variable fit_range, as Decay's.")]
    plots = [Plot("fit", "Fit", draw=draw_fit, help="Data, fit, and the fitted range (drag its edges).")]
    interactions = [RangeSelect("fit_range", plots=["fit"])]
```
In the template both `fit_range` parameters are bound to one analysis variable: dragging the range in the view refits.

| Kind | Base class | Has | Has not |
|---|---|---|---|
| reader | `tesselab.Reader` | a `P.Path`/`P.Paths` parameter, `outputs`, `run` | plots |
| compute | `tesselab.Node` | `inputs`, `params`, `outputs`, `run`, `key_results`, `actions` | plots |
| view | `tesselab.View` | `inputs`, `params`, `plots`, `interactions` | outputs, `run`, key results |

Class attributes (all optional except `run` of readers and compute nodes):

| Attribute | What |
|---|---|
| `name`, `help` | Shown in the window. |
| `inputs` | `{name: Signature}`: the data the node needs (below). |
| `params` | List of parameters (below). |
| `outputs` | Names of the Datasets `run` returns, or `{name: Signature}` to have check verify their shape. |
| `key_results` | `Result(name, label, unit, display, error, output, digits)`: single numbers shown in the card's header and saved with the analysis; read from the scalar variable `name` of the outputs. |
| `views` | Ids of views (in this module, or full ids) added with the node and connected to it. |
| `plots` | Views only: list of `Plot` (below). |
| `interactions` | Views only: cursors, crosshairs, ranges, picks, toggles that set the view's parameters from its plots (below). |
| `actions` | `Action(id, label, run, job=False)`: buttons; `run(ctx)` returns nothing, `{param: value}` (one undoable change; each value is checked by its parameter) or `("file", name, bytes)`. `job=True`: runs in a worker process (for slow actions; the window stays usable). |
| `cache` | `"memory"` (default) or `"disk"`: slow nodes keep their outputs in `tesselab/cache/` across sessions. |
| `run_on` | `"change"` (default) or `"button"`: slow jobs run only when the user presses Run. |
| `schema` + `migrate(params, from_schema)` | Raise `schema` when parameters change (renamed, new meaning) and convert old ones in `migrate`; old analysis files keep working. |
### Rules for run(inputs, params)
- **Pure**: everything comes in as arguments; it returns `{output name: xarray.Dataset}` (a DataArray is wrapped). No globals, no files written (the window watches the files nodes read: a node writing next to them would run itself again), no Qt, no window, no printing.
- `inputs.<name>` is the bound Dataset (below); an optional input that is not connected is missing (`"name" in inputs`). `params.<name>` are values in their stored units; paths are absolute.
- **Deterministic**: the same inputs and parameters give the same numbers. Random starts only through a `P.Seed` parameter.
- **Partial results are fine**: if something cannot be computed, return what you have and say why with `tesselab.diagnostic("error" | "warning" | "info", message, code="...")` (or `tesselab.warning(...)`). An exception is shown as the node's error; the window keeps working.
- **Units are SI** (m, s, J, W, OD for absorbance) in data and parameters; put the unit into the `units` attribute of coordinates and variables. Plots convert (`display`).
- **Name variables by meaning** (`wavelength`, `delay`, `absorbance`, `total_absorbed_energy`), coordinates for positions: other nodes find them by name and shape.
- Keep the computation in `analysis.py`; `run` only connects it to Tesselab.
## Inputs: signatures
A signature says what shape of data an input needs. Dimension names in `Var` are placeholders; the same placeholder means the same dimension.
```python
Signature(matrix=Var("n", "m"), row_pos=Var("n", optional=True), col_pos=Var("m"))  # a matrix and its axes
Signature(absorbance=Var("spectrum", "wavelength"))  # names of a specific kind of data
```
- Tesselab matches a signature against the outputs of the analysis: a variable whose **name** equals the signature's name is taken first; otherwise any variable (data variable or coordinate) with the right number of dimensions, numbers only, and the required `units` if given. When several fit, the user chooses in the window.
- The node sees the matched Dataset renamed to the signature's names and placeholder dimensions (the matrix transposed into the declared order); other variables of the output are kept. `ds.attrs["tesselab_names"]` and `["tesselab_dims"]` hold the original names (for axis labels).
- `Var(..., units="m")` only where the node computes physically with the unit; generic nodes (SVD) require none. `Var(..., numeric=False)` accepts text coordinates.
- `Signature(..., optional=True)`: the input may be unconnected; it is also left out when its source gives no such output (a reader with an empty file parameter returns `{}`). `Signature(..., many=True)`: a list of Datasets, one per analysis. `Signature(..., help="...")`: what the input is for (the graph window's tooltip).
- Name inputs and outputs by what the data are (`measured_spectra`, `uncombined_spectra`, `phi0_fit`), not by where they come from; the graph shows these names on the ports. Keep reading files (a node with a `P.Path` parameter, drawn with a file symbol) apart from computing.
## Parameters
`P.<Type>(name, default, *, label, help, group, advanced, optional, redraw)`. `name` is the key in the analysis file: never rename it without a `migrate` step. `redraw=True`: a plot setting; changing it redraws, `run` is not repeated. `optional=True`: `None` is allowed ("not set").

| Type | Value in the file | Notes |
|---|---|---|
| `P.Bool` | `true` | |
| `P.Int` | `5` | `min`, `max` |
| `P.Float` | `2.6e-7` | `unit`, `display` (e.g. `unit="m", display="nm"`), `min`, `max`; never NaN |
| `P.Str` | `"UC"` | `multiline` |
| `P.Choice` | `"mean"` | `options`: list or `{key: label}`; the key is stored |
| `P.Path` | `"data/blank.csv"` | relative to the project, `/` everywhere; `filter`, `folder=True`; `run` gets it absolute |
| `P.Paths` | `["a.csv"]` or `"powers/*.csv"` | a list or a glob; `run` gets a sorted list of absolute paths |
| `P.Range` | `[3.85e-7, 3.98e-7]` | start ≤ stop, both included; `unit`, `display` |
| `P.Ranges` | `[[...], [...]]` | |
| `P.Floats`, `P.Ints` | `[...]` | Ints are typed as `0-3, 7` |
| `P.Points` | `[[x, y], ...]` | picked on a plot |
| `P.Seed` | `0` | `None`: not reproducible |
| `P.Steps` | `[{"kind": "chirp", "on": true, ...}]` | `kinds=[P.Step(kind, label, params=[...])]`: an ordered list of processing steps, each switchable, with its own parameters |

Display units: SI prefixes of the stored unit (`nm`, `ps`, `mOD`, `mJ`, `mW`), `min`/`h` for `s`, `%`, and `cm-1`/`eV` for wavelengths in `m`.
## Plots (views only)
```python
Plot(id, title, draw=function, span=1, rows=1, help="", link=None, needs=(), settings=(), items="one")
```
- `draw(ctx)` returns a `PlotSpec`. `ctx.inputs` the view's bound inputs (an input whose source has no result is missing: `"name" in ctx.inputs`; draw the rest and a `note_only` where needed), `ctx.params` its parameter values, `ctx.label` (the analysis), `ctx.vars`, `ctx.diagnostics`; for `items="several"`: `ctx.others`, a list of `(label, ctx)` of the linked analyses' same view.
- What `draw` computes must be quick (an interpolation, a sum, a cut through a map): it runs in the window's thread at every redraw. Anything slow is a compute node (cached, in a worker process).
- Tesselab draws a **preview** of any reader's or compute node's output by itself (the graph window's side panel, from the data's shape): no plots are needed to check a node while building.
- `span`/`rows`: 1 or 2 tiles. `settings`: names of (usually `redraw=True`) parameters shown in a row above the plot. `needs`: output variables without which the plot shows a note. `help`: the (i) text.
- `widget=QWidgetClass` instead of `draw` (level 2): any Qt widget with `update(ctx)` (3D, editors). Use only when a description cannot say it: the export gets only a picture, and it is only as stable as the libraries it uses.
### PlotSpec
```python
PlotSpec(x=Axis("Wavelength", "m", display="nm"), y=Axis("Absorbance", "OD"),
         marks=[...], y_right=None, legend="top right", zero_line=True, grid=True, note="")
```
Several panels with shared axes (a map with profiles):
```python
PlotSpec(panels={"map": Panel(x=..., y=..., marks=[Heatmap(...)]), "right": Panel(...), "bottom": Panel(...)},
         mosaic=[["map", "right"], ["bottom", None]], share_x=[["map", "bottom"]], share_y=[["map", "right"]],
         sizes=([3, 1], [3, 1]))
```
`Axis(label, unit, display=None, scale="linear" | "log" | "linlog", linthresh=None, invert=False, limits=None)`. `display` may be a list (the user switches). `linlog`: linear within ±`linthresh` (stored units), logarithmic beyond: delays from fs to ns.

Marks (every mark takes `name=`, `style=Style(...)`, `axis="left" | "right"`; data in the axis's stored unit, numpy or xarray):

| Mark | Data | Options |
|---|---|---|
| `Line(x, y)` | | `error_y` (band), `step` |
| `Points(x, y)` | | `error_x`, `error_y`, `labels` (tool tips), `keys` (an int per point, for `Toggle`) |
| `Lines(x, y2d)` | one row per line | `colors`, `colormap` (`"default"`, `"black-red"`, `"viridis"`, `"greys"`), `labels` (tool tips; legend entries up to 12 lines), `legend=False` (tool tips only) |
| `Band(x, low, high)` | | |
| `VLine(x)`, `HLine(y)` | | `label` |
| `Region(starts, stops)` | shaded spans along x | `label` |
| `Heatmap(x, y, z)` | `z.shape == (len(x), len(y))`, cell centers | `colormap` (`"diverging"` default, symmetric around 0), `limits`, `colorbar` (its label) |
| `Text(x, y, text)` | | `coords="data" | "tile"` (0…1), `anchor` |

`Style(role, color, width, dash, symbol, size, alpha)`: `role` says what the mark is, and the user's theme gives each role its look, the same in every module (a field you set yourself wins over the role's):

| Role | What | Look in the default themes |
|---|---|---|
| `"measured"` | measured points | blue crosses (`data`, `x`, size 11) |
| `"fit"` | a fit or model line | grey dashed, width 2 |
| `"result"` | the number the analysis is after, at its place (φ₀ at x = 0) | black bar (`_`), size 14, width 2 |
| `"reference"` | an expected or literature curve | green, width 2.5 |
| `"unused"` | excluded data (ignored spectra) | grey dashed, see-through |
| `"event"` | something that happened (an interrupt of the run) | grey dashed line, width 1.5 |

Prefer a role to setting the look yourself: the user restyles all modules at once in a theme file (`~/.config/tesselab/themes/<name>.toml`: `[marks.fit]` with `color`, `width`, `dash`, `symbol`, `size`, `alpha`). `symbol` is `o`, `s`, `t`, `d`, `+`, `x` or `_`; `color` is `"#rrggbb"` or a role (`"data"`, `"fit"`, `"accent"`, `"warning"`, `"error"`, `"good"`, `"muted"`, `"ink"`, `"series:3"`). Roles follow the user's theme (light or dark); prefer them. `note_only(text)` is an empty plot that says why it is empty.
## Interactions (views only)
They write into the view's parameters (undoable, saved, shared between linked analyses); a node never handles the mouse. To reach a compute node, bind the view's parameter and the compute node's to one analysis variable in the template (`{"var": "fit_range"}` in both): the drag changes the variable, the compute node runs again.

| Interaction | Writes | |
|---|---|---|
| `Cursor(param, axis="x", plots=[...], snap=None, link=None, panel=None)` | a `Float` | a draggable line; `snap`: a variable of the view's inputs whose values it snaps to |
| `Crosshair(x=param, y=param, plots=[...], panel=None)` | two `Float`s | |
| `RangeSelect(param, plots=[...], axis="x")` | a `Range` | |
| `Pick(param, plots=[...])` | `Points` | Ctrl+click adds, Ctrl+right-click removes |
| `Toggle(param, plots=[...])` | `Ints` | a click on a point with a key (`Points(..., keys=...)`) adds the key or removes it (e.g. ignore a spectrum) |

`link`: cursors with the same key follow each other across linked analyses (e.g. `"lambda"`). `panel`: the panel name in a plot with several.
## Templates
A template is an analysis file with `{placeholders}`:
```json
{"tesselab": 1, "label": "{name}",
 "vars": {"lambda": "{lambda_spectrum}"},
 "nodes": {
   "read": {"node": "org.example.thing/read", "params": {"file": "{file}"}},
   "fit":  {"node": "org.example.thing/fit", "inputs": {"trace": "read.trace"},
            "params": {"wavelength": {"var": "lambda"}}}},
 "view": {"order": ["fit", "read"], "open": ["fit"]}}
```
- A string that is only a placeholder takes the value with its type (number, list, null). Default values: `name` (the folder's name) and `folder` (relative to the project); `fill(folder, project_root)` adds or overrides.
- An input is `"<node>.<output>"`; `{"from": ..., "bind": {"row_pos": "total_absorbed_energy"}}` fixes a choice.
- `vars` are **analysis variables**: a parameter `{"var": "lambda"}` takes the variable's value, so one cursor moves every node bound to it. They are also how a view's interactions reach the compute nodes, and where the sample's numbers live (the graph window shows them as a table).
- Put the views first in `view.order` and open them (`view.open`): they are the card's sections; the readers and compute nodes are folded under "Details".
- Paths in parameters are relative to the project folder.
- In a project, the analyses made from one template share a **graph file** (`tesselab/graphs/<template file name>.json`, below). The values filled in from placeholders are each analysis's own; every other value is the graph's, shared by all of them. So put everything that comes from the data into placeholders (files, the instrument's settings), and fixed choices into the template itself.
## Good nodes
What makes a module pleasant from the first try, learned while building the UV and TA modules. The user should not have to rename, re-cut or restyle nodes afterwards.
### Cutting the graph into nodes
- **One node, one job, named by that job.** A good split: *readers* (one per kind of file: they only read and return data as measured, with units), *corrections* (blank, background, baseline, combining), *the domain's own step* (pairing, fitting), *results* (the number the user is after), and *views* that show them. A node that reads a file and computes is two nodes; a node that computes and plots is a compute node and a view.
- **Views are what the user looks at.** Group plots into views the way the user thinks about the results (the old tool's groups, a figure of a paper), not one view per compute node: a view may take inputs from every stage (spectra from the corrections, yields from the fit, raw traces from a reader). Give a compute node that is useful on its own a default view (`views = [...]`).
- **Readers stay general.** "Agilent spectra file" reads every spectrum of the file; which one is the blank is the corrections node's business. A reader whose file parameter is empty (an optional file, e.g. a buffer) returns `{}` and reports an `info`; an optional input fed by it is then simply left out.
- **Cut where caching gains.** When a slow result does not depend on a value the user drags (a cursor), give it its own node that is not bound to that value; Tesselab then reuses it while the cursor moves (φ0 for every wavelength does not depend on λ spectrum). Conversely, do not split a computation into nodes that always change together.
- **Never pass parameters as data.** Numbers of the sample that several nodes need (thickness, volume, a sample composition, which x axis to plot against) are parameters of each of those nodes, bound to one **analysis variable** in the template. An output called `setup` or `settings` is a sign that this went wrong. The same holds for what the user sets with the mouse in a view (a λ line, a fit range, picked onsets, ignored points): a parameter of the view and of the compute node, bound to one variable.
- **Keep the user's numbers where the user looks for them.** A value that describes the sample or the measurement is an analysis variable (shown in the graph's variables table and in the header); a value that only tunes one step stays a parameter of that node.
- **Wrapping an existing package:** call its functions in the same order on the same data, so that results stay bit-identical, and prove it with a test that compares both on the sample (and a script for all of the user's data).
### Names
- **Inputs and outputs are named by what the data are**, as a scientist would say it: `measured_spectra`, `uncombined_spectra`, `spectra_with_totals`, `quantum_yields`, `phi0_fit`, `epsilon`, `rig_records`. Not by where they come from or by a vague word (`measured`, `data`, `result`, `setup`, `out2`). They are shown on the ports; keep them under about 20 characters.
- **Give every input a `help`** (`Signature(..., help="The corrected spectra, one per measuring point.")`): it is the port's tool tip.
- **Variables in the Datasets** carry their meaning in their name (`absorbance`, `delta_spectrum`, `phi0_err`) and coordinates carry `units`. Name dimensions by meaning too (`spectrum`, `step`, `wavelength`, `nucleoside`): the user sees them in tool tips.
- **Node names** are short nouns of what the node is ("Spectrum corrections", "Irradiation steps", "Power meter folder"); a reader's name says what it reads ("… file", "… folder").
### Parameters
- Stored in SI units, shown in the field's units (`unit="m", display="nm"`; `display="%"` for fractions). Label them as the field says them (`label="λ irradiation"`).
- Plot-only choices are `redraw=True` and listed in the plot's `settings` (colormap, which x axis, which spectra to show): changing them redraws without running the node again.
- A value that can be "off" is `optional=True` (empty = off), not a magic number.
### Plots
- **Match the conventions of the field and of the tool the user used before.** Ask for (or look at) the user's current plots: units (% for yields, mOD, nm), symbols (crosses for measured points), colors (spectra black → red in the order measured, unused data grey dashed, fits grey dashed, the result in black), what is written into the plot (φ₀ = (1.84 ± 0.04) % in a corner). The user wants their old plots back, not new ones.
- Marks by role (`Style(role="measured" | "fit" | "result" | "reference" | "unused" | "event")`) and colors by role (`data`, `fit`, `muted`, `ink`, `series:N`), so that the user's theme decides the look and the dark theme works; a fixed `#rrggbb` only for a meaning that must not change.
- One idea per plot, at most one legend; no legend when the colors are explained by the plot's `help` (black → red).
- Every plot has a `help`: what is drawn, what each color means, what the user can do in it (drag the red line).
- An empty plot says why and what to do: `note_only("Set the nucleosides (e.g. UC) to calculate the quantum yields.")`.
- Draggable lines (`Cursor`) are bound to the analysis variable they set, so that every plot showing it follows.
### Results and messages
- The number the user is after is a `key_result` (it is shown in the card's header and saved with the analysis), in the user's units and digits.
- When a later step cannot be done, still return what could be done (the spectra without the steps) and report why with `diagnostic("error", ...)`; a warning names what the user should set.
### Templates
- A template reproduces the user's usual analysis completely: every node connected, the sample's numbers as analysis variables filled from the data folder (`fill`), the important groups open (`view.open`), the result node first (`view.order`).
- Values that belong to one sample (file paths, the sample's numbers) are filled in from the folder; everything else is the graph's default and shared by all analyses made from the template.
## The analysis and graph files (for reference)
`tesselab/graphs/<name>.json`, the graph: `label`, `comment`, `vars` (default values), `var_pos`, `nodes` (instance name → `node` (full id), `version`, `schema`, `inputs`, `params` (the graph's values), optional `label`, optional `pos`: [x, y] in grid units of the graph window), `order`, `provenance` (`template`).

`tesselab/analyses/<label>.json`, the analysis: `label`, `comment`, `graph` (the graph file, relative to this file), `params` (instance → the values this analysis sets itself: data paths, placeholders, values changed in it), `vars` (its values), `inputs` (instance → input → links to other analyses), `results` (key results at the last save), `view` (open groups), `provenance`. An analysis without `graph` has its whole graph inside (`nodes`, `view.order`), as a template does. Links to other analyses: `"analyses/<file>.json#<node>.<output>"` (relative to the project's `tesselab/` folder), with `seen`: the fingerprint of that result at the last save. Unknown fields are kept.
## Checklist
- [ ] `tesselab check <folder>` passes without errors; warnings are understood.
- [ ] The PNGs in `check/` show what was asked: axis labels with units, sensible ranges, legends where needed.
- [ ] Variables are named by meaning, coordinates carry `units`.
- [ ] Every parameter has a `help`; risky ones are `advanced`.
- [ ] Random numbers only through a `P.Seed`.
- [ ] `sample/` is small and real; `tests/` checks a known answer if there is one.
- [ ] No Qt, files or globals in `run`; the computation is in `analysis.py`.
- [ ] The nodes follow "Good nodes": readers apart from computing, plots only in views, inputs and outputs named by what the data are and with a `help`, the sample's numbers and the mouse's settings as analysis variables, plots in the user's familiar style.
