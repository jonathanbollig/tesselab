"""
What the rig recorded about a run: steps.jsonl (one record per step, with its start as epoch time and its duration) and
config.json (the parameters of the run), next to the power folder. Only runs from the second series (data/*@2) on have them; the
older experiment.log has no dates and is not read.

Used by the viewer to show interrupts (e.g. "Extract sample for HPLC") and when the stirrer ran, and by load_data for
the clock time of every measured spectrum (spectrum_times). Nothing here changes the analysis. The times are those of the rig's clock, the same clock as the times in the power files.
"""
import json
import os
import shlex
from dataclasses import dataclass

STARTUP_MIXING = 5.0  # s; the startup step stirs 4 s, then reads the RPM for 1 s (fixed in the rig's code)


@dataclass
class Interrupt:
    """A pause in which the operator did something, e.g. took a sample out; from the prompt until Enter was pressed."""
    start: float  # epoch time [s]
    end: float
    message: str


@dataclass
class Stirring:
    """The stirrer ran from start to end (epoch times [s]). For a cycle: its number, UV time and what the rig measured
    (result: power_mw, power_std_mw, rpm, temp_c, humidity_pct)."""
    start: float
    end: float
    cycle: int | None = None  # None: the mixing before the first spectra (startup)
    uv_seconds: float | None = None
    result: dict | None = None


@dataclass
class RunRecords:
    steps: list[dict]  # the records of steps.jsonl, in order
    config: dict       # config.json: started, commit, simulated, config (the parameters at the start)

    @property
    def interrupts(self) -> list[Interrupt]:
        return [Interrupt(step["started"], step["started"] + step["duration_s"], _message(step["line"]))
                for step in self.steps if step["step"] == "interrupt"]

    @property
    def stirring(self) -> list[Stirring]:
        """When the stirrer ran. A step ends with the settling pause and the spectra (spectrometer_n spectra,
        t_spectrometer apart); the stirrer stopped just before them. Only steps that finished normally."""
        stirring, uv_before = [], 0.0
        for step in self.steps:
            if step["outcome"] != "ok" or step["step"] not in ("startup", "cycle"):
                continue
            cfg = step["config"]
            end = (step["started"] + step["duration_s"] - cfg["t_motor_settle"]
                   - cfg["spectrometer_n"] * cfg["t_spectrometer"])
            if step["step"] == "startup":
                stirring.append(Stirring(end - STARTUP_MIXING, end))
            else:
                stirring.append(Stirring(step["started"], end, step["cycle"], step["uv_seconds"] - uv_before,
                                         step["result"]))
                uv_before = step["uv_seconds"]
        return stirring

    @property
    def spectrum_times(self) -> list[float]:
        """When the rig triggered each spectrum (epoch times [s]), in order: a step ends with spectrometer_n spectra,
        t_spectrometer apart. Only steps that finished normally. The spectrometer's own collection times (in the .DSW
        file, to the second) agree within 1 s."""
        times = []
        for step in self.steps:
            if step["outcome"] != "ok" or step["step"] not in ("startup", "cycle"):
                continue
            cfg = step["config"]
            first = step["started"] + step["duration_s"] - cfg["spectrometer_n"] * cfg["t_spectrometer"]
            times += [first + k * cfg["t_spectrometer"] for k in range(cfg["spectrometer_n"])]
        return times

    @property
    def not_ok(self) -> list[dict]:
        """Steps that did not finish normally (e.g. aborted)."""
        return [step for step in self.steps if step["outcome"] != "ok"]


def _message(line) -> str:
    """"interrupt 'Extract sample for HPLC'" -> "Extract sample for HPLC"."""
    try:
        return " ".join(shlex.split(line)[1:])
    except ValueError:
        return line


def read_run_records(power_folder) -> RunRecords | None:
    """The records of the run whose power files are in power_folder (steps.jsonl and config.json in the folder above
    it); None if there are none."""
    folder = os.path.dirname(os.path.normpath(power_folder))
    steps_file, config_file = os.path.join(folder, "steps.jsonl"), os.path.join(folder, "config.json")
    if not (os.path.isfile(steps_file) and os.path.isfile(config_file)):
        return None
    with open(steps_file) as f:
        steps = [json.loads(line) for line in f if line.strip()]
    with open(config_file) as f:
        config = json.load(f)
    return RunRecords(steps, config)
