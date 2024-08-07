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

import datetime


from nomad.datamodel import EntryArchive
from nomad.metainfo import (
    Quantity,
)
from nomad.parsing import MatchingParser
from nomad.datamodel.metainfo.annotations import (
    ELNAnnotation,
)
from nomad.datamodel.data import (
    EntryData,
)

from nomad.datamodel.metainfo.basesections import (
    Activity,
)


from nomad_material_processing.utils import create_archive
from baseclasses.helper.utilities import set_sample_reference


class ParsedFile(EntryData):
    activity = Quantity(
        type=Activity,
        a_eln=ELNAnnotation(
            component='ReferenceEditQuantity',
        )
    )


class PVDPParser(MatchingParser):

    def parse(self, mainfile: str, archive: EntryArchive, logger) -> None:
        from hzb_combinatorial_libraries.schema_packages.hzb_library_package import (UnoldThermalEvaporation, UnoldXRFMeasurementLibrary, UnoldUVvisReflectionMeasurementLibrary,
                                                                                     UnoldUVvisTransmissionMeasurementLibrary, UnoldPLMeasurementLibrary, UnoldConductivityMeasurementLibrary, UnoldTRPLMeasurementLibrary)

        entry = None
        file = mainfile.split('/')[-1]

        if "pvd" in file.lower():
            entry = UnoldThermalEvaporation(log_file=file)

        if "refl20" in file:
            entry = UnoldUVvisReflectionMeasurementLibrary(data_file=file)

        if "trans20" in file:
            entry = UnoldUVvisTransmissionMeasurementLibrary(data_file=file)

        if "trpl" in file and file.endswith(".knc"):
            entry = UnoldTRPLMeasurementLibrary(data_file=file)

        if "PL" in file:
            entry = UnoldPLMeasurementLibrary(data_file=file)

        if "resist" in file.lower():
            entry = UnoldConductivityMeasurementLibrary(data_file=file)

        if "xrf" in file.lower():
            entry = UnoldXRFMeasurementLibrary(composition_file=file)
        if entry is None:
            return

        search_id = file.split("#")[0]
        set_sample_reference(archive, entry, search_id)

        entry.datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        entry.name = f"{search_id}"

        file_name = f'{file}.archive.json'
        archive.data = ParsedFile(activity=create_archive(entry, archive, file_name))
        archive.metadata.entry_name = file
