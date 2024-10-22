from nomad.config.models.plugins import AppEntryPoint

from hzb_combinatorial_libraries.apps.combinatorial_app import combinatorial_app

combinatorial_library_app = AppEntryPoint(
    name='Compinatorial Samples',
    description='Provides filters to investigate combinatorial libraries.',
    app=combinatorial_app,
)
