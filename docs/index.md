# Welcome to the `HZB-combinatorial-lab`  plugin documentation


## Introduction


### **Supported Entry Types**
This plugin supports multiple entry types, including:

1. **Library Entries:**
- Contains basic information about the smaple library, including its id, its main elements, the dimensions and the substrate used

2. **Process Entries:**
The process reference the library entry in which they are performed. This allows to track the whole history of the sample library.
- **Thermal evaporation** process. This entry logs the data of the process inclusing set and logges parameter for the sources, substrate heaters and the chamber pressure.

3. **Mapping Measurements Entries:**
The measuremnt entries reference the library entry in which they are performed. This allows to track the whole history of the sample library. At the moment mapping measurements entries include:

- **X-Ray Fluoressence Mapping**. This entry logs the data of the XRF mapping measurement, it derives the elemental composition per layer at every measured position (defined by a x, y, z grid). It also logs the derived logged thickness of the layers.
- **UV-vis Spectroscopy**. This entry logs the data of the UV-vis spectroscopy measurement, it derives the transmittance, refletance and absorbance at every measured position (defined by a x, y, z grid).
- **Steady-state Photoluminescence (PL)**. This entry logs the data of the steady-state photoluminescence measurement, at every measured position (defined by a x, y, z grid).
- **Resisitivity**. This entry logs the data of the sheet resistance measurement, at every measured position (defined by a x, y, z grid). If the thickness of tehactive layer is also logged, the resistivity is also calculated.
- **Time-resolved Photoluminescence (TRPL)**. This entry logs the data of the time-resolved photoluminescence measurement, at every measured position (defined by a x, y, z grid).


<div markdown="block" class="home-grid">
<div markdown="block">

### How-to guides

How-to guides provide step-by-step instructions for a wide range of tasks, with the overarching topics:

- [Install this plugin](how_to/install_this_plugin.md)
- [Use this plugin](how_to/use_this_plugin.md)

</div>

<div markdown="block">


### Reference

The reference [section](reference/references.md) includes a list fo the measurements supported by this plugin.

</div>
</div>
