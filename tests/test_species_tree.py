from __future__ import annotations

import pytest

from omamer.database import Database
from omamer.species_tree import SpeciesTree, SpeciesTreeError


def _taxa(newick, root=None):
    tree = SpeciesTree.from_string(newick)
    tree = tree.subtree(root if root is not None else tree.root)
    return {
        name: (parent, children, level, is_species, nspecies)
        for name, parent, children, level, is_species, nspecies in tree.taxa()
    }


def test_topology_and_species_counts():
    taxa = _taxa("(((A,B)AB,C)ABC,D)R;")
    assert taxa == {
        "R": (None, ["ABC", "D"], 0.0, False, 4),
        "ABC": ("R", ["AB", "C"], 1.0, False, 3),
        "AB": ("ABC", ["A", "B"], 2.0, False, 2),
        "A": ("AB", [], 3.0, True, 1),
        "B": ("AB", [], 3.0, True, 1),
        "C": ("ABC", [], 2.0, True, 1),
        "D": ("R", [], 1.0, True, 1),
    }


def test_children_in_file_order():
    taxa = _taxa("(D,(C,(B,A)BA)CBA)R;")
    assert taxa["R"][1] == ["D", "CBA"]
    assert taxa["BA"][1] == ["B", "A"]


def test_levels_are_branch_lengths_to_the_whole_tree_root():
    # as ete3's get_distance: missing branch lengths count as 1, the branch
    # length of the root is not counted, and the levels of a subtree are
    # measured from the root of the whole tree
    newick = "((A:0.5,B:2)AB:1.5,C)R:3;"
    taxa = _taxa(newick)
    assert {name: t[2] for name, t in taxa.items()} == {
        "R": 0.0, "AB": 1.5, "A": 2.0, "B": 3.5, "C": 1.0,
    }
    assert {name: t[2] for name, t in _taxa(newick, "AB").items()} == {
        "AB": 1.5, "A": 2.0, "B": 3.5,
    }


@pytest.mark.parametrize(
    "newick, names",
    [
        # numeric internal labels, e.g. NCBI taxon ids, are names, not
        # confidence values
        ("((9606,9598)207598,C)1;", {"1", "207598", "9606", "9598", "C"}),
        ("((A,B)'1.5e3',C)0123;", {"0123", "1.5e3", "A", "B", "C"}),
        # quoted names with spaces, dots and parentheses
        (
            "(('S. cerevisiae (strain S288c)',B)'Sacc. (x)',C)R;",
            {"R", "Sacc. (x)", "S. cerevisiae (strain S288c)", "B", "C"},
        ),
        # underscores are kept
        ("((Homo_sapiens,B)Homi_ninae,C)R;", {"R", "Homi_ninae", "Homo_sapiens", "B", "C"}),
        # comments are ignored
        ("((A,B)AB[&&NHX:x=1],C[0.9])R;", {"R", "AB", "A", "B", "C"}),
    ],
)
def test_names(newick, names):
    assert set(_taxa(newick)) == names


def test_lookup_by_exact_name():
    tree = SpeciesTree.from_string("(('S. cerevisiae (strain S288c)',B)'Sacc. (x)',C)R;")
    assert "S. cerevisiae (strain S288c)" in tree
    assert "S" not in tree
    tree.check_taxon("S. cerevisiae (strain S288c)")
    with pytest.raises(SpeciesTreeError, match="Unable to find"):
        tree.check_taxon("S")
    assert set(_taxa("(('S. cerevisiae (strain S288c)',B)'Sacc. (x)',C)R;", "Sacc. (x)")) == {
        "Sacc. (x)", "S. cerevisiae (strain S288c)", "B",
    }


@pytest.mark.parametrize(
    "newick",
    [
        "((A,B)X,(C,D)X)R;",  # two separate taxa of the same name
        "(((A,B)X,C)X,D)R;",  # a taxon nested in a taxon of the same name
        "((A,B)X,(C,X)Y)R;",  # a species with the name of an internal taxon
    ],
)
def test_duplicate_names_are_rejected(newick):
    # issue #52: duplicate names were merged into one taxon, or made a taxon
    # its own parent, so that mkdb looped forever
    with pytest.raises(SpeciesTreeError, match="must be unique.*X"):
        _taxa(newick)


def test_duplicate_names_outside_the_root_taxon_are_allowed():
    newick = "(((A,B)X,C)Keep,((D,E)X,F)Other)R;"
    assert set(_taxa(newick, "Keep")) == {"Keep", "X", "A", "B", "C"}
    with pytest.raises(SpeciesTreeError, match="Ambiguous root taxon X"):
        _taxa(newick, "X")


def test_unnamed_nodes_are_rejected():
    with pytest.raises(SpeciesTreeError, match="1 nodes .* have no name"):
        _taxa("((A,B),C)R;")
    # but they may be outside the root taxon
    assert set(_taxa("(((A,B)AB,C)ABC,(D,E))R;", "ABC")) == {"ABC", "AB", "A", "B", "C"}


def test_parents_of_the_whole_tree():
    tree = SpeciesTree.from_string("(((A,B)AB,C)ABC,D)R;")
    assert tree.parents() == {
        "R": None, "ABC": "R", "AB": "ABC", "A": "AB", "B": "AB", "C": "ABC", "D": "R",
    }


ORTHOXML_05 = """<?xml version='1.0' encoding='utf-8'?>
<orthoXML xmlns="http://orthoXML.org/2011/" origin="test" version="0.5">
  <species name="HUMAN" taxonId="4" NCBITaxId="9606">
    <database name="db" version="1"><genes><gene id="1" protId="h1" /></genes></database>
  </species>
  <species name="PANTR" taxonId="5" NCBITaxId="9598">
    <database name="db" version="1"><genes><gene id="2" protId="p1" /></genes></database>
  </species>
  <species name="MOUSE" taxonId="3" NCBITaxId="10090">
    <database name="db" version="1"><genes><gene id="3" protId="m1" /></genes></database>
  </species>
  <taxonomy>
    <taxon id="1" name="Euarchontoglires">
      <taxon id="2" name="Homininae">
        <taxon id="4" name="Homo sapiens" />
        <taxon id="5" name="Pan troglodytes" />
      </taxon>
      <taxon id="3" name="Mus musculus" />
    </taxon>
  </taxonomy>
  <groups>
    <orthologGroup id="1" taxonId="1"><geneRef id="1" /><geneRef id="3" /></orthologGroup>
  </groups>
</orthoXML>
"""


def test_taxonomy_from_orthoxml(tmp_path):
    path = tmp_path / "hogs.orthoxml"
    path.write_text(ORTHOXML_05)
    tree = SpeciesTree.from_orthoxml(path)
    taxa = {
        name: (parent, children, level, is_species, nspecies)
        for name, parent, children, level, is_species, nspecies
        in tree.subtree(tree.root).taxa()
    }
    # the species are named as in the <species> elements, which refer to
    # their taxon by taxonId
    assert taxa == {
        "Euarchontoglires": (None, ["Homininae", "MOUSE"], 0.0, False, 3),
        "Homininae": ("Euarchontoglires", ["HUMAN", "PANTR"], 1.0, False, 2),
        "HUMAN": ("Homininae", [], 2.0, True, 1),
        "PANTR": ("Homininae", [], 2.0, True, 1),
        "MOUSE": ("Euarchontoglires", [], 1.0, True, 1),
    }


def test_orthoxml_without_taxonomy(tmp_path):
    path = tmp_path / "hogs.orthoxml"
    start, end = ORTHOXML_05.index("  <taxonomy>"), ORTHOXML_05.index("  <groups>")
    path.write_text(ORTHOXML_05[:start] + ORTHOXML_05[end:])
    with pytest.raises(SpeciesTreeError, match="has no <taxonomy>"):
        SpeciesTree.from_orthoxml(path)


def test_taxonomy_table_rejects_duplicate_names(tmp_path):
    tree_path = tmp_path / "species.nwk"
    tree_path.write_text("(((A,B)X,C)X,D)R;\n")
    db = Database(tmp_path / "taxonomy.h5", root_taxon="R", mode="w")
    try:
        with pytest.raises(SpeciesTreeError, match="must be unique"):
            db.initiate_tax_tab(str(tree_path))
    finally:
        db.close()
