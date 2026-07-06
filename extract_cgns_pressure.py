"""STAR-CCM+ CGNS 压力提取工具的兼容入口。"""

if __name__ == "__main__":
    from starccm_pressure.extract_cgns_pressure import main

    raise SystemExit(main())
else:
    import sys

    from starccm_pressure import extract_cgns_pressure as _impl

    sys.modules[__name__] = _impl
