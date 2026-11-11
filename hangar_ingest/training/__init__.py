"""Leakage-safe training sample construction.

The training package rebuilds every input as it was known at a fixed knowledge
cutoff. Post-hoc corrections to work orders and calibrations are versioned and
can never travel backwards across that cutoff; when historical evidence is
missing the candidate sample is reported as not buildable instead of silently
using current state.
"""
