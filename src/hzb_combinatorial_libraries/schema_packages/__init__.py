from nomad.config.models.plugins import SchemaPackageEntryPoint


class HZBLibraryPackageEntryPoint(SchemaPackageEntryPoint):

    def load(self):
        from hzb_combinatorial_libraries.schema_packages.hzb_library_package import m_package
        return m_package


hzb_library_package = HZBLibraryPackageEntryPoint(
    name='HZBLibrary',
    description='Package for HZB Library Unold Lab',
)
