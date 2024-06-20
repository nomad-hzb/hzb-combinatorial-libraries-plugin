import json

from nomad.search import search
from nomad import files
from nomad.datamodel import EntryArchive
from nomad.app.v1.models.models import MetadataRequired
from nomad.datamodel.metainfo.basesections import CompositeSystemReference
from baseclasses.helper.utilities import search_entry_by_id, get_reference


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

#
# # # get library entry id
def get_entryid(archive, sample_id):  # give it a batch id
    # get al entries related to this batch id

    query = {'results.eln.lab_ids': sample_id}
    search_result = search(
        owner='all',
        query=query,
        user_id=archive.metadata.main_author.user_id
    )

    return search_result.data[0]["entry_id"]


def set_library_reference(archive, entry, search_id):
    search_result = search_entry_by_id(archive, entry, search_id)
    if len(search_result.data) == 1:
        data = search_result.data[0]
        upload_id, entry_id = data["upload_id"], data["entry_id"]
        if "sample" in data["entry_type"].lower() or "library" in data["entry_type"].lower():
            entry.library_reference = CompositeSystemReference(reference=get_reference(upload_id, entry_id))
        if "solution" in data["entry_type"].lower() or "ink" in data["entry_type"].lower():
            entry.library_reference = CompositeSystemReference(reference=get_reference(upload_id, entry_id))


def create_archive(entity, archive, file_name) -> str:
    import json
    from nomad.datamodel.context import ClientContext
    if isinstance(archive.m_context, ClientContext):
        return None
    if not archive.m_context.raw_path_exists(file_name):
        entity_entry = entity.m_to_dict(with_root_def=True)
        with archive.m_context.raw_file(file_name, 'w') as outfile:
            json.dump({"data": entity_entry}, outfile)
        archive.m_context.process_updated_raw_file(file_name)

    return get_reference(
        archive.metadata.upload_id,
        get_entry_id_from_file_name(file_name, archive)
    )


def get_entry_id_from_file_name(file_name, archive):
    from nomad.utils import hash
    return hash(archive.metadata.upload_id, file_name)

