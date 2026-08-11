# Answers to the 18 required audit questions

1. **Actual simulation datasets:** `metal` 200、`silicon` 200、`wumu` 1,000 solver-generated cases under `RAW_DATA_ROOT`.
2. **Actual physical experimental datasets:** `metal_10times_dlm` 375 raw DLM3000 traces；`wumu_post0_waveforms` 213 processed traces whose physical acquisition is upstream of the retained CSVs.
3. **Genuinely measured labels:** waveform voltage is measured for DLM；temperature is only filename metadata, not yet tied to a documented sensor/position.
4. **Experimental temperature meaning:** wumu unit is °C by filename, but physical meaning is unknown；metal unit and meaning are unknown. **USER CONFIRMATION REQUIRED.**
5. **Experimental full-field ground truth:** **NO**.
6. **Simulation waveform and field strictly paired:** **YES, 1,400/1,400**, by same-case artifacts and config references.
7. **Simulation and experiment operating conditions paired:** **NO；exact 0, approximate 0**.
8. **Waveform domain gap:** **CRITICAL**；group-split classifier accuracy/AUC 1.000/1.000.
9. **Is gap only incompatible acquisition contracts:** **NO conclusion possible**；contract mismatch is major, but amplitude-free relative features remain fully separable. Physics-only contribution is not identifiable.
10. **Do both retain temperature information:** **YES**, strong in simulation and moderate/confounded in experiment.
11. **Are trends cross-domain consistent:** **PARTIAL**；wumu peak time/RMS directions agree, spectral directions disagree.
12. **Is experiment within simulation support:** scalar wumu temperature range **YES** (302.15–442.15 K within 300–1500 K)；full condition support **UNKNOWN/POSSIBLY OOD**.
13. **Is weakly supervised sim-to-real defensible:** **CONDITIONALLY**, only after label semantics, H(T), acquisition metadata and spatial validation are supplied.
14. **Is condition-aware domain adaptation justified:** **Plausible and preferable to global alignment**, but currently blocked by missing condition metadata and severe mismatch.
15. **Does simulator need calibration/new cases:** **YES**, before strong sim-to-real claims.
16. **PINN components to reuse/rewrite:** reuse parsers、pairing/schema、mesh/fixed-node sampling；rewrite split/preprocessing/evaluation；scientifically replace current architecture/loss/PINN physics.
17. **New repository:** **YES LATER**, after audit acceptance and metadata confirmation；not created now.
18. **Largest scientific risk:** real 2D field is unidentifiable/unverifiable, so predictions may merely reproduce a simulation prior.
