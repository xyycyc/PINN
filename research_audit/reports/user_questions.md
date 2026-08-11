# User questions

## Q1. Experimental temperature meaning

Evidence:
- post0 filenames explicitly encode `TxxC`; metal DLM files use bare numeric names.
- no thermocouple/IR/log/config identifies the measurement operator.

Unknown:
- furnace setpoint, thermocouple, surface, average, maximum, or another quantity; metal filename unit.

Why it matters:
- determines target unit, weak loss `H(T)`, support comparison and scientific claim.

Question for user:
- What physical quantity, device and spatial location generated each filename temperature, and is metal `30…210` in °C?

## Q2. Wumu experimental acquisition contract

Evidence:
- processed CSV retains time and residual columns but not hardware configuration.

Unknown:
- specimen geometry/layers, transducer model/position, excitation, gain, coupling, raw sample source and filter/residual procedure.

Why it matters:
- currently the domain gap cannot be separated into contract mismatch and physics mismatch.

Question for user:
- Are acquisition logs or original oscilloscope files available for the wumu post0 data?

## Q3. Spatial temperature validation

Evidence:
- no measured IR field or known-coordinate thermocouple table was found; six `p1…p6` files are simulation monitor outputs.

Unknown:
- whether unindexed IR images/thermocouple records exist elsewhere.

Why it matters:
- without real spatial measurements, field reconstruction accuracy is unidentifiable.

Question for user:
- Do any calibrated IR images or thermocouple measurements with coordinates/timestamps exist outside the scanned roots?

## Q4. Simulator reproducibility

Evidence:
- configs/logs retain solver parameters and old executable/mesh paths.

Unknown:
- whether the solver source, meshes and runtime are available and runnable on the current machine.

Why it matters:
- calibration and new experiment-matched cases require rerunning the simulator.

Question for user:
- Is the heat/ultrasonic solver environment and its original mesh input still available?
