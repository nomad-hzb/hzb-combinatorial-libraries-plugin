#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from baseclasses.catalysis.catalysissample import (
    CombinatorialProperty,
    Formula,
    Thickness,
    XRayDiffraction,
)
from nomad.datamodel.data import (
    ArchiveSection,
)
from nomad.metainfo import Quantity, SubSection
from nomad_material_processing.combinatorial import CombinatorialSample


class Conductivity(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
        The conductivity of the sample.
        """,
        unit='S/m',
    )


class CarrierLifetime(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
        The lifetime of the (majority) carriers in the sample.
        """,
        unit='s',
    )


class BandGap(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
        The band gap of the sample.
        """,
        unit='eV',
    )


class Synthesis(CombinatorialProperty):
    method = Quantity(
        type=str,
        description="""
        The method used to synthesize the material.
        """,
    )
    temperature = Quantity(
        type=float,
        description="""
        The (maximum) temperature at which the material was synthesized.
        """,
        unit='K',
    )
    pressure = Quantity(
        type=float,
        description="""
        The (average) pressure at which the material was synthesized.
        """,
        unit='Pa',
    )
    atmosphere = Quantity(
        type=str,
        description="""
        The atmosphere in which the material was synthesized.
        """,
    )


class PLPeakPostion(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
      The peak position of the photoluminescence spectrum.
      """,
        unit='eV',
    )


class PLFWHM(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
      The full width at half maximum of the photoluminescence spectrum.
      """,
        unit='eV',
    )


class PLPeakArea(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
      The peak area of the photoluminescence spectrum.
      """,
    )


class PLAbsorbedPowerFlux(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
      The (assumed) absorbed power flux of the sample during the photoluminescence
      measurement.
      """,
        unit='W/m^2',
    )


class PLQY(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
        The photoluminescence quantum yield of the sample.
        """,
    )


class Photoluminescence(ArchiveSection):
    excitation_wavelength = Quantity(
        type=float,
        description="""
      The (peak) wavelength of the excitation source used during the photoluminescence
      measurement.
      """,
        unit='nm',
    )

    excitation_power = Quantity(
        type=float,
        description="""
      The power of the excitation source used during the photoluminescence
      measurement.
      """,
        unit='W',
    )

    long_pass_filter = Quantity(
        type=float,
        description="""
      The long pass filter wavelength of the excitation set up used during the photoluminescence
      measurement.
      """,
        unit='nm',
    )

    spot_size_diameter = Quantity(
        type=float,
        description="""
      The diameter of spot_size  of the excitation source used during the photoluminescence
      measurement.
      """,
        unit='m',
    )

    peak_position = SubSection(section_def=PLPeakPostion)
    fwhm = SubSection(section_def=PLFWHM)
    peak_area = SubSection(section_def=PLPeakArea)
    absorber_power_flux = SubSection(section_def=PLAbsorbedPowerFlux)
    plqy = SubSection(section_def=PLQY)


class ComplexRefractiveIndex(CombinatorialProperty):
    def derive_n_values(self):
        if self.n is not None:
            return len(self.n)
        if self.k is not None:
            return len(self.k)
        if self.photon_wavelength is not None:
            return len(self.photon_wavelength)
        else:
            return 0

    n_values = Quantity(type=int, derived=derive_n_values)

    n = Quantity(
        type=float,
        description="""
        The (real part of the) refractive index of the sample.
        """,
        shape=['n_values'],
        a_plot=[
            {
                'label': 'n data over wavelength',
                'x': 'photon_wavelength',
                'y': 'n',
                'layout': {
                    'yaxis': {'fixedrange': False},
                    'xaxis': {'fixedrange': False},
                },
            }
        ],
    )
    k = Quantity(
        type=float,
        description="""
        The attenuation coefficient of the sample.
        """,
        shape=['n_values'],
        a_plot=[
            {
                'label': 'k data over wavelength',
                'x': 'photon_wavelength',
                'y': 'k',
                'layout': {
                    'yaxis': {'fixedrange': False},
                    'xaxis': {'fixedrange': False},
                },
            }
        ],
    )
    photon_wavelength = Quantity(
        type=float,
        description="""
        The wavelength of the photons used to measure the complex refractive index.
        """,
        unit='nm',
        shape=['n_values'],
    )


class PhotovoltaicEfficiency(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
        The (theoretical) efficiency of the sample as a solar cell under AM1.5G.
        """,
    )


class PhotovoltaicShortCircuitCurrentDensity(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
        The (theoretical) short circuit current of the sample as a solar cell under
        AM1.5G.
        """,
        unit='A/m^2',
    )


class PhotovoltaicOpenCircuitVoltage(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
        The (theoretical) open circuit voltage of the sample as a solar cell under
        AM1.5G.
        """,
        unit='V',
    )


class PhotovoltaicFillFactor(CombinatorialProperty):
    value = Quantity(
        type=float,
        description="""
       The (theoretical) fill factor of the sample as a solar cell under
       AM1.5G.
       """,
    )


class Photovoltaic(ArchiveSection):
    efficiency = SubSection(section_def=PhotovoltaicEfficiency)
    jsc = SubSection(section_def=PhotovoltaicShortCircuitCurrentDensity)
    voc = SubSection(section_def=PhotovoltaicOpenCircuitVoltage)
    ff = SubSection(section_def=PhotovoltaicFillFactor)


class ThinFilmCombinatorialSample(CombinatorialSample):
    formula = SubSection(section_def=Formula)
    thickness = SubSection(section_def=Thickness)
    conductivity = SubSection(section_def=Conductivity)
    carrier_lifetime = SubSection(section_def=CarrierLifetime)
    band_gap = SubSection(section_def=BandGap)
    synthesis = SubSection(section_def=Synthesis)
    photoluminescence = SubSection(section_def=Photoluminescence)
    xray_diffraction = SubSection(section_def=XRayDiffraction)
    complex_refractive_index = SubSection(section_def=ComplexRefractiveIndex)
    photovoltaic = SubSection(section_def=Photovoltaic)
