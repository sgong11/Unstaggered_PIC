# Manuscript reproduction notes

The supplied manuscript is retained as [manuscript.pdf](manuscript.pdf).
The clean runnable package implements the four main Table 1 cases, the
projection off/on comparison, and the auxiliary spline degree-one/two comparison.
See the [main README](../README.md) for commands.

## What is specified and implemented

- TSI main dt=0.1 is run to T=60, matching the manuscript analysis interval.
  The paper describes taking this interval from a longer T=100 archive.
- New Weibel decks use the manuscript's inner iteration cap of 64. The removed
  historical decks allowed 128; production archive settings would be needed to
  reconcile that historical discrepancy.
- Main cases and the projection comparison solve particle orbits over the full
  field step. Uniform subcycling and adaptive field steps remain disabled.
- Both spline cases use identical selective refinement controls. Each field
  step starts with one particle step; a failed local solve activates the existing
  controller, doubles only failing particles' substep counts, and restarts outer
  Anderson after a layout change. Field dt stays 0.1. The allowed cap is 64,
  inherited from the supplied implementation. The paper reports only 2 and 4
  substeps actually used and does not specify the cap or all retry details.
- The spline diagnostic saves line/knot probes every 10 units of physical time,
  plus endpoints, using the supplied implementation's settings. Confirm this
  cadence against the original auxiliary archive before comparing maxima.

## Remaining limitations

1. **Full production results are not reproduced here.** Reduced numerical tests
   passed, but complete original CSV/configuration archives were not supplied.
   The reported projection RMS values, spline-knot maxima, first refinement at
   t=53.9, and long-time growth/energy statistics still need full-grid checking.
2. **Auxiliary policy provenance needs confirmation.** The manuscript does not
   fully specify the refinement cap or output cadence. Both can affect the
   reported behavior or sampled maxima. The current pair uses matched settings;
   this does not prove agreement with the historical run.
3. **OpenMP and cluster behavior remain unverified locally.** The previous
   checks used serial Apple clang. Full-grid runtime, memory, thread-count
   sensitivity, and Slurm integration need validation on the target machine.
4. **Conservation does not establish accuracy.** Finite-offset knot differences
   measure regularity, not exact-solution error. Near-zero transverse velocity
   is a longitudinal TSI diagnostic; it is not appropriate for Weibel.
5. **Data assumptions matter.** Velocity moments are mass-weighted and equal
   charge-weighted moments for these electrons because m=|q|. Full snapshots
   may be large; actual save times can differ from requested times. The paired
   helper rejects unequal or incomplete archives rather than trimming them.
6. **Inspect effective input settings.** The parser retains permissive legacy
   aliases and warnings for unknown keys. Check `_config_echo.txt`. The solver
   can overwrite reused prefixes and does not check every later disk-write
   failure; the supplied runners isolate each run in a new directory.

The original source used for the earlier full-step comparison had SHA256
`e29e44e2683d390a42e3021e8191c4632d64dfa864a57ecfa1ab9c462caa7a88`.
The root `SHA256SUMS` describes the current clean package. Historical source
copies, old inputs, job logs, and generated data were removed at cleanup.

[Validation record](VALIDATION.md) · [Code scope notes](CODE_NOTES.md)
