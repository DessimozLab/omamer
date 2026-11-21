from omamer.orthoxml_parser import OrthoxmlParser


def test_species_of_non_contiguous_gene_ids(tmp_path):
    # gene ids need not form one contiguous range per species, e.g. here
    # A has 1 and 3, B has 2 and 4
    path = tmp_path / "hogs.orthoxml"
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<orthoXML xmlns="http://orthoXML.org/2011/" origin="test" '
        'originVersion="1" version="0.3">'
        '<species name="A" NCBITaxId="1"><database name="db" version="1"><genes>'
        '<gene id="1" protId="a1"/><gene id="3" protId="a3"/>'
        "</genes></database></species>"
        '<species name="B" NCBITaxId="2"><database name="db" version="1"><genes>'
        '<gene id="2" protId="b2"/><gene id="4" protId="b4"/>'
        "</genes></database></species>"
        '<groups><orthologGroup id="1"><geneRef id="1"/><geneRef id="2"/>'
        "</orthologGroup></groups></orthoXML>"
    )
    species = OrthoxmlParser(str(path)).species
    assert [species.get_species(i).name for i in (1, 2, 3, 4)] == ["A", "B", "A", "B"]
    assert species.get_xref(3, xref_type="protId") == "a3"
