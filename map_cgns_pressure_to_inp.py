"""Compatibility entry point for CGNS pressure to Abaqus INP mapping."""

if __name__ == "__main__":
    from starccm_pressure.map_cgns_pressure_to_inp import main

    raise SystemExit(main())
else:
    import sys

    from starccm_pressure import map_cgns_pressure_to_inp as _impl

    sys.modules[__name__] = _impl
