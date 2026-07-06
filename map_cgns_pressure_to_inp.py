"""CGNS 压力到 Abaqus INP 映射工具的兼容入口。"""

if __name__ == "__main__":
    from starccm_pressure.map_cgns_pressure_to_inp import main

    raise SystemExit(main())
else:
    import sys

    from starccm_pressure import map_cgns_pressure_to_inp as _impl

    sys.modules[__name__] = _impl
