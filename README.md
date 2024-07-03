# PVD Process from HZB Unold Lab
This is an example of a PVD process from Thomas Unold's lab at the Helmholtz-Zentrum Berlin (HZB).

The example contains the raw process log `PVDProcess.csv` and an iPython notebook `generate_pvd_archive.ipynb` for creating NOMAD archives from the log.

The substances used in teh sources are created as seperate archives with the data retrieved by NOMAD from an API call to CAS.

The main process archive is `hzb_unold_lab_pvd_example.data.archive.json`.

# Installation

Can be installed with [uv](https://github.com/astral-sh/uv)  
```
uv pip install -e '.[dev]'
```

or by specificing nomads pip registry:  

```
pip install -e '.[dev]' --index-url https://gitlab.mpcdf.mpg.de/api/v4/projects/2187/packages/pypi/simple
```

# Nomad configuration for testing

You can test this plugin by configuring the `nomad.yaml` in nomad repo as below:
    
```yaml
plugins:
  entry_points:
    include: [
    "hzb_combinatorial_libraries.schema_packages:hzb_library_package"
    ]
```
