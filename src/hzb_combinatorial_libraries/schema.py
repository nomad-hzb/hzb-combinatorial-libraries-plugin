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
import os
import pandas as pd

from nomad.units import ureg
from baseclasses.solar_energy import (
    PLMeasurement,
    UVvisMeasurementLibrary, UVvisDataSimple, UVvisSingleLibraryMeasurement, UVvisProperties,
    ConductivityMeasurementLibrary, ConductivityProperties,  ConductivitySingleLibraryMeasurement, PLPropertiesLibrary, PLDataSimple, PLSingleLibraryMeasurement, PLMeasurementLibrary,
    TimeResolvedPhotoluminescenceMeasurementLibrary, TimeResolvedPhotoluminescenceSingleLibraryMeasurement, TRPLPropertiesBasic, TRPLDataSimple
)
from baseclasses.characterizations import (
    XRFLibrary, XRFSingleLibraryMeasurement, XRFProperties, XRFComposition, XRFData, XRFLayer)
from baseclasses.helper.utilities import convert_datetime, set_sample_reference
from baseclasses import (
    LibrarySample
)
from nomad.datamodel.data import EntryData
import datetime

from nomad_material_processing.physical_vapor_deposition import (
    # PVDChamberEnvironment,
    # PVDMaterialEvaporationRate,
    # PVDMaterialSource,
    # PVDPressure,
    # PVDSourcePower,
    # PVDSubstrate,
    # PVDSubstrateTemperature,
    ThermalEvaporation,
    ThermalEvaporationHeater,
    ThermalEvaporationHeaterTemperature,
    ThermalEvaporationSource,
    ThermalEvaporationStep,
)
from nomad_material_processing.utils import create_archive
from structlog.stdlib import (
    BoundLogger,
)
from nomad.metainfo import (
    Package,
    Section,
    Quantity,
)

from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    BrowserAnnotation,
)
from nomad.datamodel.metainfo.eln import (
    Substance,
)
from nomad.metainfo.metainfo import (
    Category,
)
from nomad.datamodel.data import (
    EntryDataCategory,
)

m_package = Package(name='hzb_combinatorial_libraries')

substance_translation = {
    'PbI2': 'Lead Iodide',
    'CsI': 'Cesium Iodide',
    'PbBr2': 'Lead Bromide',
    'CsBr': 'Cesium Bromide'
}


class UnoldLabCategory(EntryDataCategory):
    m_def = Category(label='HZB Unold Lab', categories=[EntryDataCategory])


class UnoldLibrary(LibrarySample, EntryData):
    m_def = Section(
        categories=[UnoldLabCategory],
        a_eln=dict(
            hide=["users", "elemental_composition", "components"]))

    qr_code = Quantity(
        type=str,
        a_eln=dict(component='FileEditQuantity'),
        a_browser=dict(adaptor='RawFileAdaptor'))

    def normalize(self, archive, logger):
        super(UnoldLibrary,
              self).normalize(archive, logger)

        with archive.m_context.raw_file(archive.metadata.mainfile) as f:
            path = os.path.dirname(f.name)

        if self.lab_id:
            import qrcode
            from PIL import ImageDraw, ImageFont
            msg = f'{self.lab_id}#'
            img = qrcode.make(msg)
            Im = ImageDraw.Draw(img)
            # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeSans.ttf", 30)

            # Add Text to an image
            Im.text((15, 15), f"{self.lab_id}")  # , font=fnt)
            qr_file_name = f"{self.lab_id}.png"
            img.save(os.path.join(path, qr_file_name), dpi=(2000, 2000))
            self.qr_code = qr_file_name


def load_XRF_txt(file):
    with open(file) as input_file:
        head = [next(input_file) for _ in range(3)]
    pos = [0]
    ns = False
    for i, c in enumerate(head[2]):
        if c != " ":
            ns = True
        if c == " " and ns and head[0][i] == " ":
            pos.append(i)
            ns = False
    pos.append(-1)
    col = []
    c_old = ''
    for i in range(len(pos)-1):
        c1 = head[0][pos[i]:pos[i+1]].strip() if head[0][pos[i]:pos[i+1]].strip() else c_old
        c2 = head[1][pos[i]:pos[i+1]].strip()
        col.append((c1, c2))
        c_old = c1
    if "," in head[2]:
        composition_data = pd.read_csv(file, names=col, header=None, skiprows=2, sep="\s{2,}", decimal=",", index_col=0)
    else:
        composition_data = pd.read_csv(file, names=col, header=None, skiprows=2, sep="\s{2,}", decimal=".", index_col=0)

    return composition_data


class UnoldXRFMeasurementLibrary(XRFLibrary, EntryData):
    m_def = Section(
        label='Unold lab XRF Measurement Library',
        categories=[UnoldLabCategory],
        a_eln=dict(hide=['instruments', 'steps', 'results', 'lab_id'],
                   properties=dict(
            order=[
                "name",
            ]))
    )

    def normalize(self, archive, logger):

        with archive.m_context.raw_file(archive.metadata.mainfile) as f:
            path = os.path.dirname(f.name)
            file_name = os.path.basename(f.name)
            if not self.samples:
                set_sample_reference(archive, self, "_".join(file_name.split("_")[0:4]).strip("#"))

        if self.samples and self.samples[0].lab_id:
            search_key = self.samples[0].lab_id
            # find data
            for item in os.listdir(path):
                if not os.path.isdir(os.path.join(path, item)):
                    continue
                if not item.startswith(f"{search_key}#"):
                    continue
                self.data_folder = item
            # find images
            images = []
            for item in os.listdir(path):
                if not os.path.isfile(os.path.join(path, item)):
                    continue
                if not (item.startswith(f"{search_key}#") and item.endswith(".bmp")):
                    continue
                images.append(item)
            self.images = images

        if self.composition_file and self.data_folder:
            file_path = os.path.join(path, self.composition_file)
            try:
                with open(file_path) as input_file:
                    pass
            except:
                file_path = os.path.join(path, self.data_folder, self.composition_file)
            measurements = []

            data_folder = os.path.join(path, self.data_folder)

            from baseclasses.helper.file_parser.xrf_spx_parser import read as xrf_read
            files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith(".spx")]
            files.sort()

            spectra, energy, measurement_rows, positions_array, position_axes, (len_x, len_y, order_letter) = xrf_read(
                files)
            self.datetime = convert_datetime(
                measurement_rows[0]["DateTime"], datetime_format="%Y-%m-%dT%H:%M:%S.%f", utc=False)
            self.energy = energy
            composition_data = load_XRF_txt(file_path)
            material_name = ''
            for i, spectrum in enumerate(spectra):
                measurement_row = composition_data.loc[os.path.splitext(os.path.basename(files[i]))[0]]
                layer_data = {}
                for v in measurement_row.items():
                    if v[0][0] not in layer_data:
                        layer_data.update({v[0][0]: {}})
                    if "Thick" in v[0][1]:
                        layer_data.update({v[0][0]: {"thickness": v[1]}})
                        continue
                    if "%" not in v[0][1]:
                        continue
                    if "composition" not in layer_data[v[0][0]]:
                        layer_data[v[0][0]].update({"composition": []})
                    if v[0][1] not in material_name:
                        material_name += f"{v[0][0]}:{v[0][1]},"
                    layer_data[v[0][0]]["composition"].append(XRFComposition(amount=v[1]))
                    # layer_data[v[0][0]]["composition"].append(XRFComposition(name=v[0][1], amount=v[1]))

                layers = []
                for key, layer in layer_data.items():
                    layers.append(XRFLayer(
                        layer=key,
                        composition=layer.get("composition", None),
                        thickness=layer.get("thickness", None)
                    ))

                measurements.append(XRFSingleLibraryMeasurement(
                    data_file=[os.path.basename(os.path.join(self.data_folder, files[i]))],
                    position_x=position_axes[0][i % len_x],  # positions_array[0, i],
                    position_y=position_axes[1][i // len_x],  # positions_array[1, i],
                    layer=layers,
                    name=f"{position_axes[0][i % len_x]},{position_axes[1][i // len_x]}")
                )
            self.measurements = measurements
            self.material_names = material_name
        super(UnoldXRFMeasurementLibrary, self).normalize(archive, logger)


class UnoldUVvisReflectionMeasurementLibrary(UVvisMeasurementLibrary, EntryData):
    m_def = Section(
        labels='Unold lab UVvis Reflection Measurement Library',
        categories=[UnoldLabCategory],
        a_eln=dict(hide=['instruments', 'steps', 'results', 'lab_id'],
                   properties=dict(
            order=[
                "name",
            ])),
        a_plot=[
            {
                'x': 'wavelength', 'y': 'measurements/:/data/intensity', 'layout': {
                    'yaxis': {
                        "range": [0, 1],
                        "fixedrange": False}, 'xaxis': {
                        "fixedrange": False}}, "config": {
                            "scrollZoom": True, 'staticPlot': False, }}]
    )

    def normalize(self, archive, logger):

        key = "refl"
        dark_key = "dark"
        reference_key = "mirror"
        with archive.m_context.raw_file(archive.metadata.mainfile) as f:
            path = os.path.dirname(f.name)
            file_name = os.path.basename(f.name)
            if not self.samples:
                set_sample_reference(archive, self, "_".join(file_name.split("_")[0:4]).strip("#"))

        if not self.reference_file:
            for file in os.listdir(path):
                if reference_key not in file or key not in file:
                    continue
                self.reference_file = file

        if not self.dark_file:
            for file in os.listdir(path):
                if dark_key not in file or key not in file:
                    continue
                self.dark_file = file

        if self.data_file and self.reference_file and self.dark_file:
            measurements = []

            from baseclasses.helper.file_parser.uvvis_parser import read_uvvis
            md, df = read_uvvis(os.path.join(path, self.data_file),
                                os.path.join(path, self.reference_file),
                                os.path.join(path, self.dark_file))
            self.datetime = convert_datetime(md["Date_Time"], datetime_format="%Y_%m_%d_%H%M", utc=False)

            if not self.samples:
                set_sample_reference(archive, self, md["Sample_ID"].strip("#"))
            if self.properties is None:
                self.properties = UVvisProperties(integration_time=md['integration time'].split(" ")[0].strip()
                                                  * ureg(md['integration time'].split(" ")[1].strip()),
                                                  spot_size=md['spot size'].split(" ")[0].strip()
                                                  * ureg(md['spot size'].split(" ")[1].strip()))
            self.wavelength = df.columns[4:]
            for i, row in df.iterrows():
                data = UVvisDataSimple(intensity=row[df.columns[4:]])

                measurements.append(UVvisSingleLibraryMeasurement(
                    position_x=row["x"],
                    position_y=row["y"],
                    position_z=row["z"],
                    data=data,
                    name=f"{row['x']},{row['y']},{row['z']}"),
                )
            self.measurements = measurements
        super(UnoldUVvisReflectionMeasurementLibrary, self).normalize(archive, logger)


class UnoldTRPLMeasurementLibrary(TimeResolvedPhotoluminescenceMeasurementLibrary, EntryData):
    m_def = Section(
        labels='Unold lab TRPL Measurement Library',
        categories=[UnoldLabCategory],
        a_eln=dict(hide=['instruments', 'steps', 'results', 'lab_id'],
                   properties=dict(
            order=[
                "name",
            ]))
    )

    def normalize(self, archive, logger):
        with archive.m_context.raw_file(archive.metadata.mainfile) as f:
            path = os.path.dirname(f.name)
            file_name = os.path.basename(f.name)
            if not self.samples:
                set_sample_reference(archive, self, "_".join(file_name.split("_")[0:4]).strip("#"))
        if self.data_file:
            import xarray as xr
            data = xr.load_dataset(os.path.join(path, self.data_file))
            # self.time = data.trpl_t.values * ureg(data.trpl_t.units)
            self.properties = TRPLPropertiesBasic(
                repetition_rate=data.trpl_repetition_rate.values*ureg(data.trpl_repetition_rate.units),
                laser_power=data.trpl_power.values*ureg(data.trpl_power.units),
                excitation_peak_wavelength=data.trpl_excitation_wavelength.values *
                ureg(data.trpl_excitation_wavelength.units),
                detection_wavelength=data.trpl_detection_wavelength.values*ureg(data.trpl_detection_wavelength.units),
                spotsize=data.trpl_counts.spot_size_um2 * ureg("um**2"),
                excitation_attenuation_filter=data.trpl_counts.attenuation,
                integration_time=data.trpl_counts.integration_time_s * ureg("s")

            )
            self.datetime = convert_datetime(data.datetime.start_datetime,
                                             datetime_format="%Y-%m-%dT%H:%M:%S.%f", utc=False)
            # measurements = []

            # for i, pos_x in enumerate(data.x.values):
            #     for j, pos_y in enumerate(data.y.values):
            #         trpl_data = TRPLDataSimple(counts=data.trpl_counts[i, j, 0, 0, 0, 0, :].values)

            #         measurements.append(TimeResolvedPhotoluminescenceSingleLibraryMeasurement(
            #             position_x=pos_x * ureg(data.x.units),
            #             position_y=pos_y*ureg(data.y.units),
            #             data=trpl_data,
            #             name=f"{pos_x},{pos_x}"),
            #         )
            # self.measurements = measurements

        super(UnoldTRPLMeasurementLibrary, self).normalize(archive, logger)


class UnoldUVvisTransmissionMeasurementLibrary(UVvisMeasurementLibrary, EntryData):
    m_def = Section(
        labels='Unold lab UVvis Transmission Measurement Library',
        categories=[UnoldLabCategory],
        a_eln=dict(hide=['instruments', 'steps', 'results', 'lab_id'],
                   properties=dict(
            order=[
                "name",
            ])),
        a_plot=[
            {
                'x': 'wavelength', 'y': 'measurements/:/data/intensity', 'layout': {
                    'yaxis': {
                        "fixedrange": False}, 'xaxis': {
                        "fixedrange": False}}, "config": {
                            "scrollZoom": True, 'staticPlot': False, }}]
    )

    def normalize(self, archive, logger):

        key = "trans"
        dark_key = "dark"
        reference_key = "light"
        with archive.m_context.raw_file(archive.metadata.mainfile) as f:
            path = os.path.dirname(f.name)
            file_name = os.path.basename(f.name)
            if not self.samples:
                set_sample_reference(archive, self, "_".join(file_name.split("_")[0:4]).strip("#"))

        if not self.reference_file:
            for file in os.listdir(path):
                if reference_key not in file or key not in file:
                    continue
                self.reference_file = file

        if not self.dark_file:
            for file in os.listdir(path):
                if dark_key not in file or key not in file:
                    continue
                self.dark_file = file

        if self.data_file and self.reference_file and self.dark_file:
            measurements = []

            from baseclasses.helper.file_parser.uvvis_parser import read_uvvis
            md, df = read_uvvis(os.path.join(path, self.data_file),
                                os.path.join(path, self.reference_file),
                                os.path.join(path, self.dark_file))
            self.datetime = convert_datetime(md["Date_Time"], datetime_format="%Y_%m_%d_%H%M", utc=False)
            if not self.samples:
                set_sample_reference(archive, self, md["Sample_ID"].strip("#"))
            if self.properties is None:
                self.properties = UVvisProperties(integration_time=md['integration time'].split(" ")[0].strip()
                                                  * ureg(md['integration time'].split(" ")[1].strip()),
                                                  spot_size=md['spot size'].split(" ")[0].strip()
                                                  * ureg(md['spot size'].split(" ")[1].strip()))
            self.wavelength = df.columns[4:]
            for i, row in df.iterrows():
                data = UVvisDataSimple(intensity=row[df.columns[4:]])

                measurements.append(UVvisSingleLibraryMeasurement(
                    position_x=row["x"],
                    position_y=row["y"],
                    position_z=row["z"],
                    data=data,
                    name=f"{row['x']},{row['y']},{row['z']}"),
                )
            self.measurements = measurements
        super(UnoldUVvisTransmissionMeasurementLibrary,
              self).normalize(archive, logger)


class UnoldPLMeasurementLibrary(PLMeasurementLibrary, EntryData):
    m_def = Section(
        label='Unold lab PL Measurement Library',
        categories=[UnoldLabCategory],
        a_eln=dict(hide=['instruments', 'steps', 'results', 'lab_id'],
                   properties=dict(
            order=[
                "name",
            ])),
        a_plot=[
            {
                'x': 'wavelength', 'y': 'measurements/:/data/intensity', 'layout': {
                    'yaxis': {
                        "fixedrange": False}, 'xaxis': {
                        "fixedrange": False}}, "config": {
                            "scrollZoom": True, 'staticPlot': False, }}]
    )

    def normalize(self, archive, logger):
        with archive.m_context.raw_file(archive.metadata.mainfile) as f:
            path = os.path.dirname(f.name)
            file_name = os.path.basename(f.name)
            if not self.samples:
                set_sample_reference(archive, self, "_".join(file_name.split("_")[0:4]).strip("#"))
        if self.data_file:
            measurements = []

            from baseclasses.helper.file_parser.pl_parser import read_file_pl_unold
            md, df = read_file_pl_unold(os.path.join(path, self.data_file))
            self.datetime = convert_datetime(md["Date_Time"], datetime_format="%Y_%m_%d_%H%M", utc=False)
            if not self.samples:
                set_sample_reference(archive, self, md["Sample_ID"].strip("#"))
            if self.properties is None:
                self.properties = PLPropertiesLibrary(integration_time=md['integration time'].split(" ")[0].strip()
                                                      * ureg(md['integration time'].split(" ")[1].strip()),
                                                      spot_size=md['spot size'].split(" ")[0].strip(),
                                                      # * ureg(md['spot size'].split(" ")[1].strip()),
                                                      long_pass_filter=md['long pass filter'].split(" ")[0].strip()
                                                      * ureg(md['long pass filter'].split(" ")[1].strip()),
                                                      laser_wavelength=md['laser wavelength'].split(" ")[0].strip()
                                                      * ureg(md['laser wavelength'].split(" ")[1].strip()))
            self.wavelength = df.columns[6:]
            for i, row in df.iterrows():
                data = PLDataSimple(intensity=row[df.columns[6:]])

                measurements.append(PLSingleLibraryMeasurement(
                    position_x=row["x"],
                    position_y=row["y"],
                    position_z=row["z"],
                    data=data,
                    name=f"{row['x']},{row['y']},{row['z']}"),
                )
            self.measurements = measurements

        super(UnoldPLMeasurementLibrary,
              self).normalize(archive, logger)


class UnoldConductivityMeasurementLibrary(ConductivityMeasurementLibrary, EntryData):
    m_def = Section(
        label='Unold lab Conductivity Measurement Library',
        categories=[UnoldLabCategory],
        a_eln=dict(hide=['instruments', 'steps', 'results', 'lab_id'],
                   properties=dict(
            order=[
                "name",
            ])),
    )

    def normalize(self, archive, logger):

        with archive.m_context.raw_file(archive.metadata.mainfile) as f:
            path = os.path.dirname(f.name)
            file_name = os.path.basename(f.name)
            if not self.samples:
                set_sample_reference(archive, self, "_".join(file_name.split("_")[0:4]).strip("#"))

        if self.data_file:
            measurements = []

            from baseclasses.helper.file_parser.conductivity_parser import read_conductivity
            md, df = read_conductivity(os.path.join(path, self.data_file))
            self.datetime = convert_datetime(md["Date_Time"], datetime_format="%Y_%m_%d_%H%M", utc=False)
            if not self.samples:
                set_sample_reference(archive, self, md["Sample_ID"].strip("#"))
            if self.properties is None:
                print(float(md['integration_time'].split(" ")[0].strip()), md['integration_time'].split(" ")[1].strip())
                print(float(md['Configuration'].split(" ")[0].strip()), md['Configuration'].split(" ")[1].strip())
                self.properties = ConductivityProperties(integration_time=float(md['integration_time'].split(" ")[0].strip())
                                                         * ureg(md['integration_time'].split(" ")[1].strip()),
                                                         configuration=float(md['Configuration'].split(" ")[0].strip())
                                                         * ureg(md['Configuration'].split(" ")[1].strip()))

            for i, row in df.iterrows():
                measurements.append(ConductivitySingleLibraryMeasurement(
                    position_x=row["x"],
                    position_y=row["y"],
                    position_z=row["z"],
                    conductivity=row["resistance"]*ureg("Gohm"),
                    name=f"{row['x']},{row['y']},{row['z']}"),
                )
            self.measurements = measurements

        super(UnoldConductivityMeasurementLibrary,
              self).normalize(archive, logger)


class Unold_Lab_Substance(Substance, EntryData):
    pass


class UnoldThermalEvaporation(ThermalEvaporation, EntryData):
    '''
    Class autogenerated from yaml schema.
    '''
    m_def = Section(
        label='Unold Lab Thermal Evaporation',
        categories=[UnoldLabCategory],
        links=["http://purl.obolibrary.org/obo/CHMO_0001360"],
        # a_plot=[
        #     dict(
        #         x='steps/:/sources/:/material_source/rate/process_time',
        #         y='steps/:/sources/:/material_source/rate/rate',
        #     ),
        #     dict(
        #         x='steps/:/sources/:/evaporation_source/temperature/process_time',
        #         y='steps/:/sources/:/evaporation_source/temperature/temperature',
        #     ),
        #     dict(
        #         x='steps/:/environment/pressure/process_time',
        #         y='steps/:/environment/pressure/pressure',
        #         layout=dict(
        #             yaxis=dict(
        #                 type='log',
        #             ),
        #         ),
        #     ),
        # ],
    )
    log_file = Quantity(
        type=str,
        description='''
        The log file generated by the PVD software.
        ''',
        a_browser=BrowserAnnotation(
            adaptor='RawFileAdaptor'
        ),
        a_eln=ELNAnnotation(
            component='FileEditQuantity'
        ),
    )

    def normalize(self, archive, logger: BoundLogger) -> None:
        '''
        The normalizer for the `HZBUnoldLabThermalEvaporation` class. Will generate and
        fill the `steps` attribute using the `log_file`.

        Args:
            archive (EntryArchive): The archive containing the section that is being
            normalized.
            logger (BoundLogger): A structlog logger.
        '''
        with archive.m_context.raw_file(archive.metadata.mainfile) as f:
            path = os.path.dirname(f.name)
            file_name = os.path.basename(f.name)
            if not self.samples:
                set_sample_reference(archive, self, "_".join(file_name.split("_")[0:4]).strip("#"))
        # if self.log_file:
        #     import pandas as pd
        #     import numpy as np
        #     with archive.m_context.raw_file(self.log_file, 'r') as fh:
        #         line = fh.readline().strip()
        #         metadata = {}
        #         while line.startswith('#'):
        #             if ':' in line:
        #                 key = line.split(':')[0][1:].strip()
        #                 value = str.join(':', line.split(':')[1:]).strip()
        #                 metadata[key] = value
        #             line = fh.readline().strip()
        #         df = pd.read_csv(fh, sep='\t')
        #     self.datetime = datetime.datetime.strptime(
        #         f'{metadata["Date"]}T{df["Time"].values[0]}',
        #         r'%Y/%m/%dT%H:%M:%S',
        #     )
        #     self.end_time = datetime.datetime.strptime(
        #         f'{metadata["Date"]}T{df["Time"].values[-1]}',
        #         r'%Y/%m/%dT%H:%M:%S',
        #     )
        #     self.name = f'PVD-{metadata["process ID"]}'
        #     self.location = 'Berlin, Germany'
        #     self.lab_id = f'HZB_{metadata["operator"]}_{self.datetime.strftime(r"%Y%m%d")}_PVD-{metadata["process ID"]}'

        #     source_materials = {column[0]: column.split()[2] for column in df.columns if column[-1:] == 'T'}

        #     qcms = ['QCM1_1', 'QCM1_2', 'QCM2_1', 'QCM2_2']
        #     qcms_source_number = {df[qcm[:-2]+" FILMNAM"+qcm[-2:]].values[0]: qcm for qcm in qcms}
        #     try:
        #         qcms_ordered = [qcms_source_number[int(source)] for source in source_materials]
        #     except KeyError:
        #         raise ValueError("Film names do not match source names.")
        #     shutters = [f'{qcm[:-2]} SHTSRC{qcm[-2:]}' for qcm in qcms_ordered]
        #     start_times = []
        #     for shutter in shutters:
        #         switch_times = df.loc[df[shutter].diff() != 0, 'Process Time in seconds'].values
        #         for time in switch_times:
        #             if not any(abs(t - time) < 5 for t in start_times):
        #                 start_times.append(time)
        #     start_times.append(df.iloc[-1, 1])
        #     substances = {
        #         source_nr: create_archive(
        #             entity=Unold_Lab_Substance(
        #                 name=substance_translation.get(
        #                     source_materials[source_nr],
        #                     source_materials[source_nr]
        #                 ),
        #             ),
        #             archive=archive,
        #             file_name=f'{source_materials[source_nr]}_substance.archive.json',
        #         ) for source_nr in source_materials
        #     }
        #     steps = []
        #     depositions = 0
        #     for idx, time in enumerate(start_times[:-1]):
        #         step = df.loc[
        #             (time <= df['Process Time in seconds'])
        #             & (df['Process Time in seconds'] < start_times[idx + 1])
        #         ]
        #         if step.loc[:, shutters].mode().any().any():
        #             depositions += 1
        #             name = f'deposition {depositions}'
        #         elif idx == 0:
        #             name = 'pre'
        #         else:
        #             name = 'post'
        #         sources = []
        #         for source_nr in source_materials:
        #             source = f'{source_nr} - {source_materials[source_nr]}'
        #             material_source = PVDMaterialSource(
        #                 material=substances[source_nr],
        #                 rate=PVDMaterialEvaporationRate(
        #                     rate=1e-6 * step[f'{source} PV'],
        #                     process_time=step['Process Time in seconds'],
        #                     measurement_type='Quartz Crystal Microbalance',
        #                 ),
        #             )
        #             evaporation_source = ThermalEvaporationHeater(
        #                 temperature=ThermalEvaporationHeaterTemperature(
        #                     temperature=step[f'{source} T'] + 273.15,
        #                     process_time=step['Process Time in seconds'],
        #                 ),
        #                 power=PVDSourcePower(
        #                     power=step[f'{source} Aout'],
        #                     process_time=step['Process Time in seconds']
        #                 ),
        #             )
        #             thermal_evaporation_source = ThermalEvaporationSource(
        #                 name=source_materials[source_nr],
        #                 material_source=material_source,
        #                 evaporation_source=evaporation_source,
        #             )
        #             sources.append(thermal_evaporation_source)
        #         substrate = PVDSubstrate(
        #             substrate=None,  # TODO: Add substrate
        #             temperature=PVDSubstrateTemperature(
        #                 temperature=step['Substrate PV'] + 273.15,
        #                 process_time=step['Process Time in seconds'],
        #                 measurement_type='Heater thermocouple',
        #             ),
        #             heater='Resistive element',
        #             distance_to_source=[
        #                 np.linalg.norm(np.array((41.54e-3, 26.06e-3, 201.12e-3)))
        #             ] * 4,
        #         )
        #         environment = PVDChamberEnvironment(
        #             pressure=PVDPressure(
        #                 pressure=step['Vacuum Pressure2'] * 1e2,
        #                 process_time=step['Process Time in seconds'],
        #             ),
        #         )
        #         step = ThermalEvaporationStep(
        #             name=name,
        #             creates_new_thin_film=step.loc[:, shutters].mode().any().any(),
        #             duration=start_times[idx + 1] - time,
        #             sources=sources,
        #             substrate=[substrate],
        #             environment=environment,
        #         )
        #         steps.append(step)
        #     self.steps = steps

        super(UnoldThermalEvaporation, self).normalize(archive, logger)


m_package.__init_metainfo__()
