import json

from nomad.search import search
from nomad import files
from nomad.datamodel import EntryArchive
from nomad.app.v1.models.models import MetadataRequired
# get a specific entry data
# hzb_combinatorial_libraries.schema.UnoldPLMeasurementLibrary


def search_data(archive, sample_id, entry_type):
    from nomad.search import search
    entry_id = get_entryid(archive, sample_id)
    q = {
            "quantities:all": [
                "metadata.entry_references.target_reference"
            ],
            "entry_references.target_entry_id:all": [
                entry_id
            ]
    }

    search_result = search(
        owner='all',
        query=q,
        user_id=archive.metadata.main_author.user_id)

    data = []
    for res in search_result.data:
        if res["entry_type"] != entry_type:
            continue
        with files.UploadFiles.get(upload_id=res["upload_id"]).read_archive(entry_id=res["entry_id"]) as archive:
            entry_id = res["entry_id"]
            data.append(archive[entry_id]["data"])

    return data


# get library entry id
def get_entryid(archive, sample_id):  # give it a batch id
    # get al entries related to this batch id

    query = {'results.eln.lab_ids': sample_id}
    search_result = search(
        owner='all',
        query=query,
        user_id=archive.metadata.main_author.user_id)

    return search_result.data[0]["entry_id"]

