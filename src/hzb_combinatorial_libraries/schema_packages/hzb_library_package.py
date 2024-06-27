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
import numpy as np
import pandas as pd

from nomad.units import ureg
from baseclasses.solar_energy import (
    UVvisMeasurementLibrary, UVvisDataSimple, UVvisSingleLibraryMeasurement, UVvisProperties,
    ConductivityMeasurementLibrary, ConductivityProperties, ConductivitySingleLibraryMeasurement, PLPropertiesLibrary,
    PLDataSimple, PLSingleLibraryMeasurement, PLMeasurementLibrary,
    TimeResolvedPhotoluminescenceMeasurementLibrary, TimeResolvedPhotoluminescenceSingleLibraryMeasurement,
    TRPLPropertiesBasic, TRPLDataSimple
)
from baseclasses.characterizations import (
    XRFLibrary, XRFSingleLibraryMeasurement, XRFProperties, XRFComposition, XRFData, XRFLayer)
from baseclasses.helper.utilities import convert_datetime, set_sample_reference
from baseclasses import (
    LibrarySample
)
from nomad.datamodel.results import BandGap
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.results import Properties, ElectronicProperties
from nomad_material_processing.combinatorial import ContinuousCombiSample, CombinatorialSamplePosition
from nomad.datamodel.data import EntryData
import datetime
from hzb_combinatorial_libraries.schema_packages.utils import set_library_reference
from hzb_combinatorial_libraries.schema_packages.pixel_processing.pl_processing import pl_data_processing
from hzb_combinatorial_libraries.schema_packages.pixel_processing.conductivity_processing import conductivity_data_processing
from hzb_combinatorial_libraries.schema_packages.pixel_processing.bilinear_interpolation import bilinear_interpolation
from hzb_combinatorial_libraries.schema_packages.pixel_processing.xrf_processing import xrf_data_processing
from nomad_material_processing.combinatorial import CombinatorialSample
# from nomad_material_processing.physical_vapor_deposition import (
# PVDChamberEnvironment,
# PVDMaterialEvaporationRate,
# PVDMaterialSource,
# PVDPressure,
# PVDSourcePower,
# PVDSubstrate,
# PVDSubstrateTemperature,
# )

from nomad_material_processing.vapor_deposition import (
    ChamberEnvironment,
    Pressure,
    Temperature,
)
from nomad_material_processing.vapor_deposition.pvd import (
    ImpingingFlux,
    SourcePower,
    PVDSampleParameters,
)
from nomad_material_processing.vapor_deposition.pvd.thermal import (
    ThermalEvaporation,
    ThermalEvaporationHeater,
    ThermalEvaporationHeaterTemperature,
    ThermalEvaporationSource,
    ThermalEvaporationStep,
)

from nomad.datamodel.metainfo.basesections import (
    PureSubstanceComponent,
    PubChemPureSubstanceSection
)

from structlog.stdlib import (
    BoundLogger,
)
from nomad.metainfo import (
    Package,
    Section,
    Quantity, SubSection,
    SchemaPackage
)

from nomad.datamodel.metainfo.basesections import (
    CompositeSystemReference,
    ElementalComposition,
)

from nomad.datamodel.metainfo.annotations import ELNComponentEnum
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
    BrowserAnnotation,
)

from nomad.metainfo.metainfo import (
    Category,
)
from nomad.datamodel.data import (
    EntryDataCategory,
)
from nomad.search import search
from .utils import (
    create_archive,
)
from nomad import files
from baseclasses.wet_chemical_deposition.wet_chemical_deposition import rewrite_json_recursively


m_package = SchemaPackage()


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

    generate_pixel = Quantity(
        type=bool,
        default=False,
        a_eln=dict(component='ButtonEditQuantity')
    )

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
            root_dir = os.path.dirname(os.path.abspath(__file__))
            fnt = ImageFont.truetype(os.path.join(root_dir, "fonts/OpenSans-VariableFont_wdth,wght.ttf"), 23)
            # Add Text to an image
            Im.text((5, 5), f"{self.lab_id}", font=fnt)
            qr_file_name = f"{self.lab_id}.png"
            img.save(os.path.join(path, qr_file_name), dpi=(2000, 2000))
            self.qr_code = qr_file_name

        if not self.generate_pixel:
            return

        # reset back the generate pixel to False
        # note that reprocessing will not regenerate the pixels, need to set generate_pixel to True manually
        rewrite_json_recursively(archive, "generate_pixel", False)
        self.generate_pixel = False

        entry_id = archive.metadata.entry_id

        search_result = search(
            owner="all",
            query={'entry_references.target_entry_id': entry_id},
            user_id=archive.metadata.main_author.user_id,
        )

        # get data
        pl_data = None
        conductivity_data = None
        xrf_data = None
        for res in search_result.data:
            with files.UploadFiles.get(upload_id=res["upload_id"]).read_archive(entry_id=res["entry_id"]) as archive_dict:
                entry_id = res["entry_id"]
                entry_data = archive_dict[entry_id]["data"]

                if entry_data["m_def"] == "hzb_combinatorial_libraries.schema_packages.hzb_library_package.UnoldPLMeasurementLibrary":
                    pl_data = entry_data
                elif entry_data["m_def"] == "hzb_combinatorial_libraries.schema_packages.hzb_library_package.UnoldConductivityMeasurementLibrary":
                    conductivity_data = entry_data
                elif entry_data["m_def"] == "hzb_combinatorial_libraries.schema_packages.hzb_library_package.UnoldXRFMeasurementLibrary":
                    xrf_data = entry_data

        # get processed data
        pl_df = None
        conductivity_df = None
        xrf_df = None

        if pl_data:
            pl_df = pl_data_processing(pl_data)
        if conductivity_data:
            conductivity_df = conductivity_data_processing(conductivity_data)
        if xrf_data:
            xrf_df = xrf_data_processing(xrf_data)

        # use xrf data (x,y) as the position for the generated pixel, if no xrf data, not generating pixels
        pixels = []
        if xrf_data:
            x = np.array(xrf_df["position_x"])
            y = np.array(xrf_df["position_y"])[::-1]
            x_quantity = ureg.Quantity(x, 'millimeter').to('meter').magnitude
            y_quantity = ureg.Quantity(y, 'millimeter').to('meter').magnitude

            # create pixels
            # for i in range(0, len(x)):
            for i in range(0, 20):
                p = Pixel()
                p.properties = PixelProperty()
                p.properties.thickness = xrf_df["thickness"][i]

                for j, amt in enumerate(xrf_df["amount"][i]):
                    elemental_composition = ElementalComposition(
                        element=xrf_df["elements"][i][j+1],
                        mass_fraction=amt,
                    )

                    p.elemental_composition.append(elemental_composition)

                p.position = CombinatorialSamplePosition(x=x_quantity[i], y=y_quantity[i])
                p.lab_id = f"{self.lab_id}:Pixel-{i}"

                pixels.append(p)

        if pl_data:
            x_pl = np.array(pl_df["position_x"])
            y_pl = np.array(pl_df["position_y"])[::-1]
            x_quantity_pl = ureg.Quantity(x_pl, 'millimeter').to('meter').magnitude # defaultDisplayUnit mm, but the unit for x and y is meter
            y_quantity_pl = ureg.Quantity(y_pl, 'millimeter').to('meter').magnitude

            voc = np.array(pl_df["voc"])
            plqy = np.array(pl_df["integral_pl"])/100
            fwhm = np.array(pl_df["FWHM"])

            for pixel in pixels:
                voc_i = bilinear_interpolation(x_quantity_pl, y_quantity_pl, voc, pixel.position.x.magnitude, pixel.position.y.magnitude)
                plqy_i = bilinear_interpolation(x_quantity_pl, y_quantity_pl, plqy, pixel.position.x.magnitude, pixel.position.y.magnitude)
                fwhm_i = bilinear_interpolation(x_quantity_pl, y_quantity_pl, fwhm, pixel.position.x.magnitude, pixel.position.y.magnitude)

                print(f"voc: {voc_i}, plqy: {plqy_i}, fwhm: {fwhm_i}")
                pixel.properties.implied_voc = voc_i
                pixel.properties.PLQY = plqy_i
                pixel.properties.FWHM = fwhm_i

        if conductivity_data:
            x_conductivity = np.array(conductivity_df["position_x"])
            y_conductivity = np.array(conductivity_df['position_y'])[::-1]
            conductivity = np.array(conductivity_df["conductivity"])

            for pixel in pixels:
                con = bilinear_interpolation(x_conductivity, y_conductivity, conductivity, pixel.position.x.magnitude, pixel.position.y.magnitude)
                pixel.properties.conductivity = con

        for pixel in pixels:
            create_archive(pixel, archive, f'{str(pixel.lab_id)}.archive.json')





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
    for i in range(len(pos) - 1):
        c1 = head[0][pos[i]:pos[i + 1]].strip() if head[0][pos[i]:pos[i + 1]].strip() else c_old
        c2 = head[1][pos[i]:pos[i + 1]].strip()
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
                if not item.startswith(f"{search_key}"):
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

        data_folder = os.path.join(path, self.data_folder if self.data_folder else '')
        if self.composition_file and self.data_folder and "Messung.spx" in os.listdir(data_folder):
            file_path = os.path.join(path, self.composition_file)
            try:
                with open(file_path) as input_file:
                    pass
            except:
                file_path = os.path.join(path, self.data_folder, self.composition_file)
            measurements = []

            from hzb_combinatorial_libraries.schema_packages.file_parser.xrf_spx_parser import read as xrf_read
            files = [os.path.join(data_folder, file) for file in os.listdir(data_folder)
                     if file.endswith(".spx") and not file.lower() == "messung.spx"]
            files.sort()

            _, energy, measurement_rows, positions_array, _, _ = xrf_read(
                [os.path.join(data_folder, "Messung.spx")])

            self.datetime = convert_datetime(
                measurement_rows[0]["DateTime"], datetime_format="%Y-%m-%dT%H:%M:%S.%f", utc=False)
            self.energy = energy
            if file_path.endswith(".txt"):
                composition_data = load_XRF_txt(file_path)
            else:
                composition_data = pd.read_excel(os.path.join(path, self.composition_file), header=[0, 1], index_col=0)
            material_name = ''
            for i, file in enumerate(files):
                _, _, _, ar, _, _ = xrf_read([file])
                measurement_row = composition_data.loc[os.path.splitext(os.path.basename(file))[0]]
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
                    position_x=positions_array[0][0] - ar[0][0],  # positions_array[0, i],
                    position_y=positions_array[1][0] - ar[1][0],  # positions_array[1, i],
                    layer=layers,
                    name=f"{round(positions_array[0][0] - ar[0][0], 5)},{round(positions_array[1][0] - ar[1][0], 5)}")
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

            from hzb_combinatorial_libraries.schema_packages.file_parser.uvvis_parser import read_uvvis
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
                repetition_rate=data.trpl_repetition_rate.values * ureg(data.trpl_repetition_rate.units),
                laser_power=data.trpl_power.values * ureg(data.trpl_power.units),
                excitation_peak_wavelength=data.trpl_excitation_wavelength.values *
                ureg(data.trpl_excitation_wavelength.units),
                detection_wavelength=data.trpl_detection_wavelength.values * ureg(data.trpl_detection_wavelength.units),
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

            from hzb_combinatorial_libraries.schema_packages.file_parser.uvvis_parser import read_uvvis
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

            from hzb_combinatorial_libraries.schema_packages.file_parser.pl_parser import read_file_pl_unold
            md, df = read_file_pl_unold(os.path.join(path, self.data_file))
            self.datetime = convert_datetime(md["Date_Time"], datetime_format="%Y_%m_%d_%H%M", utc=False)
            if not self.samples:
                set_sample_reference(archive, self, md["Sample_ID"].strip("#"))
            if self.properties is None:
                self.properties = PLPropertiesLibrary(integration_time=md['integration time'].split(" ")[0].strip()
                                                      * ureg(
                    md['integration time'].split(" ")[1].strip()),
                    spot_size=md['spot size'].split(" ")[0].strip(),
                    # * ureg(md['spot size'].split(" ")[1].strip()),
                    long_pass_filter=md['long pass filter'].split(" ")[0].strip()
                    * ureg(
                                                          md['long pass filter'].split(" ")[1].strip()),
                    laser_wavelength=md['laser wavelength'].split(" ")[0].strip()
                    * ureg(
                                                          md['laser wavelength'].split(" ")[1].strip()))
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

            from hzb_combinatorial_libraries.schema_packages.file_parser.conductivity_parser import read_conductivity
            md, df = read_conductivity(os.path.join(path, self.data_file))
            self.datetime = convert_datetime(md["Date_Time"], datetime_format="%Y_%m_%d_%H%M", utc=False)
            if not self.samples:
                set_sample_reference(archive, self, md["Sample_ID"].strip("#"))
            if self.properties is None:
                print(float(md['integration_time'].split(" ")[0].strip()), md['integration_time'].split(" ")[1].strip())
                print(float(md['Configuration'].split(" ")[0].strip()), md['Configuration'].split(" ")[1].strip())
                self.properties = ConductivityProperties(
                    integration_time=float(md['integration_time'].split(" ")[0].strip())
                    * ureg(md['integration_time'].split(" ")[1].strip()),
                    configuration=float(md['Configuration'].split(" ")[0].strip())
                    * ureg(md['Configuration'].split(" ")[1].strip()))

            for i, row in df.iterrows():
                measurements.append(ConductivitySingleLibraryMeasurement(
                    position_x=row["x"],
                    position_y=row["y"],
                    position_z=row["z"],
                    conductivity=row["resistance"] * ureg("Gohm"),
                    name=f"{row['x']},{row['y']},{row['z']}"),
                )
            self.measurements = measurements

        super(UnoldConductivityMeasurementLibrary,
              self).normalize(archive, logger)


default_source_mapping = {
    "PbI2": 1,
    "CsI": 2,
    "CsBr": 3,
    "SnI2": 5,
}


class UnoldThermalEvaporation(ThermalEvaporation, EntryData):
    """
    Class autogenerated from yaml schema.
    """

    m_def = Section(
        categories=[UnoldLabCategory],
        label="Thermal Evaporation Process",
        links=["http://purl.obolibrary.org/obo/CHMO_0001360"],
        a_plot=[
            dict(
                label="Impinging flux",
                x="steps/:/sources/:/impinging_flux/:/time",
                y="steps/:/sources/:/impinging_flux/:/value",
            ),
            dict(
                label="Temperature",
                x="steps/:/sources/:/vapor_source/temperature/time",
                y="steps/:/sources/:/vapor_source/temperature/value",
            ),
            dict(
                label="Pressure",
                x="steps/:/environment/pressure/time",
                y="steps/:/environment/pressure/value",
                layout=dict(
                    yaxis=dict(
                        type="log",
                    ),
                ),
            ),
        ],
    )
    log_file = Quantity(
        type=str,
        description="""
        The log file generated by the PVD software.
        """,
        a_browser=BrowserAnnotation(adaptor="RawFileAdaptor"),
        a_eln=ELNAnnotation(component="FileEditQuantity"),
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
        if self.log_file:
            import pandas as pd
            import numpy as np

            with archive.m_context.raw_file(self.log_file, "r") as fh:
                line = fh.readline().strip()
                metadata = {}
                while line.startswith("#"):
                    if ":" in line:
                        key = line.split(":")[0][1:].strip()
                        value = str.join(":", line.split(":")[1:]).strip()
                        metadata[key] = value
                    line = fh.readline().strip()
                df = pd.read_csv(fh, sep="\t")
                if (df["Process Time in seconds"] == 0).all():
                    df["Process Time in seconds"] = np.arange(df.shape[0])
            self.datetime = datetime.datetime.strptime(
                f'{metadata["Date"]}T{df["Time"].values[0]}',
                r"%Y/%m/%dT%H:%M:%S",
            )
            self.end_time = datetime.datetime.strptime(
                f'{metadata["Date"]}T{df["Time"].values[-1]}',
                r"%Y/%m/%dT%H:%M:%S",
            )
            self.name = f'PVD-{metadata["process ID"]}'
            self.location = "Berlin, Germany"
            self.lab_id = f'HZB_{metadata["operator"]}_{self.datetime.strftime(r"%Y%m%d")}_PVD-{metadata["process ID"]}'

            source_materials = {
                column[0]: column.split()[2]
                for column in df.columns
                if column[-1:] == "T"
            }

            qcms = ["QCM1_1", "QCM1_2", "QCM2_1", "QCM2_2"]
            qcms_source_number = {
                df[qcm[:-2] + " FILMNAM" + qcm[-2:]].values[0]: qcm for qcm in qcms
            }
            try:
                qcms_ordered = [
                    qcms_source_number[default_source_mapping[material]] for source, material in
                    source_materials.items()
                ]
            except KeyError:
                raise ValueError("Film names do not match source names.")
            shutters = [f"{qcm[:-2]} SHTSRC{qcm[-2:]}" for qcm in qcms_ordered]
            start_times = []
            for shutter in shutters:
                switch_times = df.loc[
                    df[shutter].diff() != 0, "Process Time in seconds"
                ].values
                for time in switch_times:
                    if not any(abs(t - time) < 5 for t in start_times):
                        start_times.append(time)
            start_times.append(df.iloc[-1, 1])
            steps = []
            depositions = 0
            for idx, time in enumerate(start_times[:-1]):
                step = df.loc[
                    (time <= df["Process Time in seconds"])
                    & (df["Process Time in seconds"] < start_times[idx + 1])
                ]

                if step.loc[:, shutters].mode().any().any():
                    depositions += 1
                    name = f"deposition {depositions}"
                elif idx == 0:
                    name = "pre"
                else:
                    name = "post"
                sources = []
                for source_nr in source_materials:
                    source = f"{source_nr} - {source_materials[source_nr]}"
                    vapour_source = ThermalEvaporationHeater(
                        temperature=ThermalEvaporationHeaterTemperature(
                            value=step[f"{source} T"] + 273.15,
                            time=step["Process Time in seconds"],
                        ),
                        power=SourcePower(
                            value=step[f"{source} Aout"],
                            time=step["Process Time in seconds"],
                        ),
                    )
                    material = [
                        PureSubstanceComponent(
                            pure_substance=PubChemPureSubstanceSection(
                                molecular_formula=source_materials[source_nr],
                            )
                        )
                    ]
                    impinging_flux = ImpingingFlux(
                        value=1e-6 * step[f"{source} PV"],
                        set_value=1e-6 * step[f"{source} TSP"],
                        time=step["Process Time in seconds"],
                        set_time=step["Process Time in seconds"],
                        measurement_type="Quartz Crystal Microbalance",
                    )
                    thermal_evaporation_source = ThermalEvaporationSource(
                        name=source_materials[source_nr],
                        material=material,
                        impinging_flux=[impinging_flux],
                        vapor_source=vapour_source,
                    )
                    sources.append(thermal_evaporation_source)
                substrate = PVDSampleParameters(
                    substrate=None,  # TODO: Add substrate
                    substrate_temperature=Temperature(
                        value=step["Substrate PV"] + 273.15,
                        set_value=step["Substrate TSP"] + 273.15,
                        time=step["Process Time in seconds"],
                        set_time=step["Process Time in seconds"],
                        measurement_type="Heater thermocouple",
                    ),
                    heater="Resistive element",
                    distance_to_source=[
                        np.linalg.norm(np.array((41.54e-3, 26.06e-3, 201.12e-3)))
                    ]
                    * 4,
                )
                environment = ChamberEnvironment(
                    pressure=Pressure(
                        value=step["Vacuum Pressure2"] * 1e2,
                        time=step["Process Time in seconds"],
                    ),
                )
                step = ThermalEvaporationStep(
                    name=name,
                    creates_new_thin_film=step.loc[:, shutters].mode().any().any(),
                    duration=start_times[idx + 1] - time,
                    sources=sources,
                    sample_parameters=[substrate],
                    environment=environment,
                )
                steps.append(step)
            self.steps = steps

        super(UnoldThermalEvaporation, self).normalize(archive, logger)


class PixelProperty(EntryData):
    m_def = Section(
        categories=[UnoldLabCategory],
        label='UnoldPixelProperty'
    )
    thickness = Quantity(
        type=np.dtype(np.float64),
        unit='cm',
        shape=[],
        description="""
            The thickness of the pixel.
            """,
        a_eln=dict(
            component='NumberEditQuantity',
            defaultDisplayUnit='mm',
        )
    )
    conductivity = Quantity(
        type=np.dtype(np.float64),
        unit='S/cm',
        shape=[],
        description="""
                The conductivity of the Pixel.
                """,
        a_eln=dict(
            component='NumberEditQuantity',
            defaultDisplayUnit='S/cm',
        )
    )
    bandgap = Quantity(
        type=np.dtype(np.float64),
        unit='eV',
        description='Band gap value in eV.',
        a_eln=dict(
            component='NumberEditQuantity',
            defaultDisplayUnit='eV',
        )
    )

    PLQY = Quantity(
        type=np.dtype(np.float64),
        description='Energy integrated value of the PL spectrum.',
        a_eln=dict(component='NumberEditQuantity')
    )

    implied_voc = Quantity(
        type=np.dtype(np.float64),
        unit='eV',
        description='Estimated open circuit voltage based on PL measurements.',
        a_eln=dict(
            component='NumberEditQuantity',
            defaultDisplayUnit='eV',
        )
    )

    FWHM = Quantity(
        type=np.dtype(np.float64),
        unit='eV',
        description='FWHM based on PL measurements.',
        a_eln=dict(
            component='NumberEditQuantity',
            defaultDisplayUnit='eV'
        )
    )


class Pixel(ContinuousCombiSample, EntryData, ArchiveSection):
    m_def = Section(
        categories=[UnoldLabCategory],
        label='UnoldPixel'
    )
    properties = SubSection(
        section_def=PixelProperty,
        description='''
          The properties of the pixel.
          ''',
    )
    library_reference = SubSection(
        section_def=CompositeSystemReference,
        description='''
          The samples refer to the library ID.
          ''',
    )

    def normalize(self, archive, logger):
        super(ContinuousCombiSample, self).normalize(archive, logger)

        # self.components for xrf, check htem how to do it, and add element to results.materials.elements
        if self.lab_id:
            id = self.lab_id.split(':')[0].strip()
            self.name = self.lab_id

            set_library_reference(archive, self, id)
        else:
            raise ValueError("Pixel Lab ID is missing")

        if self.properties and self.properties.bandgap:
            bg = BandGap(value=np.float64(self.properties.bandgap) * ureg('eV'))
            if not archive.results.properties:
                archive.results.properties = Properties()
            if not archive.results.properties.electronic:
                archive.results.properties.electronic = ElectronicProperties(band_gap=[bg])


m_package.__init_metainfo__()
