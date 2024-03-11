# PVD Process from HZB Unold Lab
This is an example of a PVD process from Thomas Unold's lab at the Helmholtz-Zentrum Berlin (HZB).

The example contains the raw process log `PVDProcess.csv` and an iPython notebook `generate_pvd_archive.ipynb` for creating NOMAD archives from the log.

The substances used in teh sources are created as seperate archives with the data retrieved by NOMAD from an API call to CAS.

The main process archive is `hzb_unold_lab_pvd_example.data.archive.json`.


# test

You can test this plugin by configuring the `nomad.yaml` in nomad repo as below:
    
```yaml
plugins:
  include:
     - 'parsers/hzb-combinatorial-libraries-plugin'
  options:
     parsers/hzb-combinatorial-libraries-plugin:
       python_package: hzb_combinatorial_libraries
```
