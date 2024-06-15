from nomad.config.models.plugins import ParserEntryPoint


class HZBLibraryParserEntryPoint(ParserEntryPoint):

    def load(self):
        from hzb_combinatorial_libraries.parsers.hzb_library_parser import PVDPParser
        return PVDPParser(**self.dict())


hzb_library_parser = HZBLibraryParserEntryPoint(
    name='HZBLibraryParserEntryPoint',
    description='Parser for Unold Library Lab HZB files',
    mainfile_name_re='^.*(pvd.*|PVD.*|PL.*|refl20.*|trans20.*|(R|r)esist.*)\.(t|c)sv|.*(xrf.*\.(txt|xlsx))|.*(trpl.*\.knc)$',
    mainfile_mime_re='(application|text)/.*'
)
