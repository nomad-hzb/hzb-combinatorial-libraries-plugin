import yaml
from nomad.config.models.ui import (
    App,
    Column,
    Columns,
    FilterMenu,
    FilterMenus,
    Filters,
)

combinatorial_app = App(
    # Label of the App
    label='HZB Combinatorial libraries',
    # Path used in the URL, must be unique
    path='combinatorialhzb',
    # Used to categorize apps in the explore menu
    category='High-Throughput Experimentation',
    # Brief description used in the app menu
    description="""
    Explore the HZB combinatorial samples.
    """,
    # Longer description that can also use markdown
    readme="""
    Explore combinatorial librearies and samples.
    """,
    # Controls the available search filters. If you want to filter by
    # quantities in a schema package, you need to load the schema package
    # explicitly here. Note that you can use a glob syntax to load the
    # entire package, or just a single schema from a package.
    filters=Filters(
        include=[
            '*#hzb_combinatorial_libraries.schema_packages.hzb_library_package.UnoldSample',
        ]
    ),
    # Controls which columns are shown in the results table
    columns=Columns(
        selected=[
            # 'data.lab_id#nomad_htem_database.schema_packages.htem_package.HTEMLibrary',
            # 'data.band_gap.value#nomad_htem_database.schema_packages.htem_package.HTEMSample',
            # 'data.thickness.value#nomad_htem_database.schema_packages.htem_package.HTEMSample',
            'results.material.elements',
            'entry_type',
            # 'data.lab_id#nomad_material_processing.combinatorial.ThinFilmCombinatorialSample'
        ],
        options={
            'entry_type': Column(label='Entry type', align='left'),
            'entry_name': Column(label='Name', align='left'),
            'entry_create_time': Column(label='Entry time', align='left'),
            'authors': Column(label='Authors', align='left'),
            'upload_name': Column(label='Upload name', align='left'),
            # 'data.lab_id#nomad_htem_database.schema_packages.htem_package.HTEMLibrary': Column(  # noqa: E501
            #     label='Library ID', align='left'
            # ),
            'results.material.elements': Column(label='Elements', align='left'),
            # 'data.band_gap.value#nomad_htem_database.schema_packages.htem_package.HTEMSample': Column(  # noqa: E501
            #     label='Bandgap', align='left'
            # ),
            # 'data.thickness.value#nomad_htem_database.schema_packages.htem_package.HTEMSample': Column(  # noqa: E501
            #     label='Thickness', align='left', unit='nm'
            # ),
            # 'data.lab_id#nomad_material_processing.combinatorial.ThinFilmCombinatorialSample': Column(label='Sample ID', align='left'),  # noqa: E501
        },
    ),
    # Dictionary of search filters that are always enabled for queries made
    # within this app. This is especially important to narrow down the
    # results to the wanted subset. Any available search filter can be
    # targeted here. This example makes sure that only entries that use
    # MySchema are included.
    filters_locked={
        'results.eln.sections': [
            # 'HTEMSample',
            'UnoldSample',
        ]
    },
    # Controls the filter menus shown on the left
    filter_menus=FilterMenus(
        options={
            'material': FilterMenu(label='Material', level=0),
            'elements': FilterMenu(label='Elements / Formula', level=1, size='xl'),
            'eln': FilterMenu(label='Electronic Lab Notebook', level=0),
            'custom_quantities': FilterMenu(
                label='User Defined Quantities', level=0, size='l'
            ),
            'author': FilterMenu(label='Author / Origin / Dataset', level=0, size='m'),
            'metadata': FilterMenu(label='Visibility / IDs / Schema', level=0),
            'optimade': FilterMenu(label='Optimade', level=0, size='m'),
        }
    ),
    # Controls the default dashboard shown in the search interface
    dashboard=yaml.safe_load(
        """
    widgets:
      - type: periodictable
        scale: linear
        search_quantity: results.material.elements
        layout:
          xxl:
            minH: 3
            minW: 3
            h: 9
            w: 12
            y: 0
            x: 0
          xl:
            minH: 3
            minW: 3
            h: 9
            w: 12
            y: 0
            x: 0
          lg:
            minH: 3
            minW: 3
            h: 7
            w: 10
            y: 0
            x: 0
          md:
            minH: 3
            minW: 3
            h: 6
            w: 8
            y: 0
            x: 0
          sm:
            minH: 3
            minW: 3
            h: 6
            w: 8
            y: 0
            x: 0
      - type: terms
        show_input: true
        scale: linear
        search_quantity: results.eln.lab_ids
        layout:
          xxl:
            minH: 3
            minW: 3
            h: 9
            w: 5
            y: 0
            x: 12
          xl:
            minH: 3
            minW: 3
            h: 9
            w: 5
            y: 0
            x: 12
          lg:
            minH: 3
            minW: 3
            h: 7
            w: 6
            y: 0
            x: 10
          md:
            minH: 3
            minW: 3
            h: 6
            w: 4
            y: 0
            x: 8
          sm:
            minH: 3
            minW: 3
            h: 6
            w: 4
            y: 0
            x: 8
"""
    ),
)
