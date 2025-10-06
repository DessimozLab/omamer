"""
    HOGPROP - Propagation of Gene Ontology (GO) annotations through
    Hierarchical Orthologous Groups (HOGs) from the OMA project.

    (C) 2024-2025 Nikolai Romashchenko <nikolai.romashchenko@unil.ch>
    (C) 2015-2023 Alex Warwick Vesztrocy <alex@warwickvesztrocy.co.uk>

    This file is part of HOGPROP. It contains a module for parsing an
    OrthoXML file which represents a set of Hierarchical Orthologous
    Groups (HOGs).

    HOGPROP is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    HOGPROP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with HOGPROP.  If not, see <http://www.gnu.org/licenses/>.
"""
from collections import defaultdict
from copy import copy
from lxml import etree
from tqdm import tqdm
from functools import lru_cache
from property_manager import lazy_property
import codecs
import numpy as np
import pandas as pd
import pickle
import os
import tables
import re


from ._utils import auto_open


# Define namespaces
NS = {
    "OrthoXML": "http://orthoXML.org/2011/",
    "re": "http://exslt.org/regular-expressions",
}
for ns in NS.items():
    etree.register_namespace(*ns)


# Hard-code current OMA ID length
OMA_ID_LEN = 5


# Exceptions
class DistanceFittingException(Exception):
    pass


class UnknownSpeciesException(Exception):
    pass


class UnknownXrefTypeException(Exception):
    pass


class GeneIDException(Exception):
    pass


# Helper functions
def is_evolutionary_node(el):
    """
    Returns true if the node is either orthologGroup or paralogGroup.
    """
    return is_orthologGroup_node(el) or is_paralogGroup_node(el)


def is_orthologGroup_node(el):
    """
    Returns true if the node is of type orthologGroup.
    """
    return is_tag_name(el, "orthologGroup")


def is_paralogGroup_node(el):
    """
    Returns true if the node is of type paralogGroup.
    """
    return is_tag_name(el, "paralogGroup")


def is_geneRef_node(el):
    """
    Returns true if the node is of type geneRef.
    """
    return is_tag_name(el, "geneRef")


def is_tag_name(el, tag_names, namespace=None):
    """
    Returns true if the node has the tag name defined.
    """
    el_tag_name = tag_name(el.tag, namespace)

    if not isinstance(tag_names, (list, set)):
        return el_tag_name == tag_names
    else:
        return el_tag_name in set(tag_names)


def tag_name(name, namespace="OrthoXML"):
    """
    Returns the name of an OrthoXML tag name. This is used for adding
    elements to the XML, externally.
    """
    if namespace is not None:
        return etree.QName(NS[namespace], name).text
    else:
        # Else return localname (i.e., don't explicitly give a namespace)
        return etree.QName(name).localname


def delete_node(node):
    """
    Cleans up a node.
    """
    node.getparent().remove(node)


def delete_nodes(nodes):
    """
    Deletes a list of nodes.
    """
    for node in nodes:
        delete_node(node)


# HOG level IDs
class HOGLevelIDs(object):
    """
    HOG level IDs
    """

    def __init__(self, hog_parser, oma_db=None):
        self.hog_parser = hog_parser
        if oma_db is not None:
            try:
                import tables
            except ImportError:
                raise ImportError("PyTables is required for this function.")
            self.oma_db = tables.open_file(oma_db, mode="r")
        else:
            self.oma_db = None

        self.__i = 0
        self.__hog_tab = {}
        self.release_char = ""
        if self.oma_db is not None:
            try:
                self.oma_db.get_node_attr("/", "oma_release_char")
            except AttributeError:
                pass

    def close(self):
        if self.oma_db is not None:
            self.oma_db.close()

    def get_map(self, hog):
        """
        Get the mapping for a single family
        """
        if self.oma_db is None:
            return self._get_map_xml(hog)
        else:
            return self._get_map_omadb(hog)

    def _get_map_xml(self, hog):
        # this is more memory intensive (we cache the results)
        # -- generate m for the hog
        def ortholog_info(node):
            loft_id = ".".join(node.attrib["og"].split(".")[1:])
            hog_level = hog.get_properties(node)["TaxRange"]
            i = self.__i
            self.__i += 1
            return ((loft_id, hog_level), i)

        if hog.id in self.__hog_tab:
            return self.__hog_tab[hog.id]

        x = ortholog_info(hog.struct)
        m = {x[0]: x[1]}
        for n in filter(is_orthologGroup_node, hog.struct.findall(".//*")):
            x = ortholog_info(n)
            m[x[0]] = x[1]

        self.__hog_tab[hog.id] = m
        return m

    def _get_map_omadb(self, hog):
        return {
            (
                b".".join(r["ID"].split(b".")[1:]).decode("ascii"),
                r["Level"].decode("ascii"),
            ): r.nrow
            for r in self.oma_db.root.HogLevel.where("Fam == {}".format(hog.fam))
        }

    def save_hog_table(self, db):
        """
        Saves the HOG table generated.
        Note: this is converted so that it is in the same format as /HogLevel
        in the pyoma DB
        """
        try:
            import tables  # optional
        except:
            raise ImportError("PyTables is required for this function.")

        def generate():
            # generate the table
            for fam, fam_map in self.__hog_tab.items():
                for (loft_id, hog_level), i in fam_map.items():
                    loft_id = "HOG:{}{:07d}".format(self.release_char, fam) + (
                        ".{}".format(loft_id) if len(loft_id) > 0 else ""
                    )
                    yield (fam, loft_id, hog_level, i)

        # generate the hog list and sort by the offset that we assigned
        x = list(map(lambda x: x[:-1], sorted(list(generate()), key=lambda x: x[-1])))

        m1 = max(map(lambda y: len(y[1]), x))
        m2 = max(map(lambda y: len(y[2]), x))

        class HOGsTable(tables.IsDescription):
            # Similar to pyoma.browser.tablefmt.HOGsTable
            Fam = tables.Int32Col(pos=1)
            ID = tables.StringCol(m1, pos=2)
            Level = tables.StringCol(m2, pos=3)

        # create table, similar to that in OMA DB
        tab = db.create_table(
            "/",
            "HogLevel",
            description=HOGsTable,
            title="similar to /HogLevel in pyomadb",
        )
        # write the table
        tab.append(x)

        # index as in pyomadb
        tab.colinstances["Fam"].create_csindex()
        tab.colinstances["ID"].create_csindex()
        tab.colinstances["Level"].create_csindex()


# HOG Parsing classes
class AllSpecies(object):
    def __init__(self, hog_parser, no_xrefs=False, is_oma=False):
        """
        Initialises species list with a dictionary
        """
        # Backup
        self.is_oma = is_oma

        # Initialise species table
        self._species = {
            species.name: species for species in hog_parser.iter_species(no_xrefs, is_oma)
        }

        # Setup mappings for those that exist:
        self._species_codes = {}
        self._species_ncbi = {}
        self._xrefs = defaultdict(dict)

        for species in self._species.values():
            # Species code
            if species.species_code is not None:
                self._species_codes[species.species_code] = species
            # NCBI TaxID
            if species.NCBITaxID > 0:
                self._species_ncbi[species.NCBITaxID] = species
            # Gene XRefs
            for gene_id, gene in species.genes.items():
                for xref_type, xrefs in gene.items():
                    for xref in xrefs.split("|"):
                        self._xrefs[xref_type][xref.strip(" ")] = gene_id

    def clear_xref(self):
        """
        Remove species object XRefs.
        """
        for sp in self._species.values():
            sp.genes.clear()
            sp.genes = {}

        for xref_type in self._xrefs:
            self._xrefs[xref_type].clear()
        self._xrefs.clear()
        self._xrefs = defaultdict(dict)

    def __iter__(self):
        """
        Iterate through species.
        """
        yield from self._species.values()

    def __getitem__(self, sp):
        """
        Gets and returns a species object. Tries based on name first, then
        attempts species code and then NCBI.
        """
        if sp in self._species:
            return self._species[sp]
        elif sp in self._species_codes:
            return self._species_codes[sp]
        elif sp in self._species_ncbi:
            return self._species_ncbi[sp]

    @lazy_property
    def _species_table(self):
        species = list(self._species.keys())
        max_ids = np.zeros((len(species),), dtype=np.uint64)
        for i, sp in enumerate(species):
            max_ids[i] = self._species[sp].max_id

        return (species, max_ids)

    def get_species(self, ids):
        """
        Gets a species (name) from a gene ID, based on ID range.
        Makes assumption that all gene ids are contiguous integers in the header file.
        This might need to be replaced for more general OrthoXML.
        NOTE: this might not work with non-OMA. Unsure of standard for this.
        """
        species = self._species_table[0]
        max_ids = self._species_table[1]

        i = np.searchsorted(max_ids, ids)

        try:
            if type(i) is np.ndarray:
                return list(map(lambda j: self._species[species[j]], i))
            else:
                return self._species[species[i]]

        except IndexError:
            for j, ix in enumerate(i):
                if ix > len(species):
                    raise UnknownSpeciesException(
                        "Can't work out which "
                        "species {:d} comes from.".format(int(ids[j]))
                    )

    def resolve_xref(self, xref_id, xref_type=None, _extra=None, as_list=None):
        """
        Resolve an xref mapping query.
        """
        if self.is_oma and hasattr(self, "_extra_xrefs") and (_extra is not False):
            x = list(self._resolve_extra_xref(xref_id))
            if as_list:
                return x
            elif len(x) == 1:
                return x[0]
            elif len(x) > 1:
                return x
            else:
                return None
        else:
            r = self._resolve_xref(xref_id, xref_type)
            return r if not as_list else ([r] if r is not None else [])

    def _resolve_xref(self, xref_id, xref_type=None):
        """
        Resolve an xref mapping query.
        """
        xref_type = "protId" if xref_type is None else xref_type
        if self.is_oma:
            # OMA *browser release* only. Therefore, check protId.
            # Check if correct format and return correct entry number if so.
            if len(xref_id) > 5:
                sp_code = xref_id[:5]
                if sp_code in self._species_codes:
                    try:
                        offset = int(xref_id[5:]) - 1
                        enum = self._species_codes[sp_code].min_id + offset
                        if enum <= self._species_codes[sp_code].max_id:
                            return enum
                    except:
                        pass

        # Check that we have that xref_type and the xref_id before returning
        if xref_type in self._xrefs and xref_id in self._xrefs[xref_type]:
            # Return it.
            return self._xrefs[xref_type][xref_id]
        elif not self.is_oma and xref_type not in self._xrefs:
            # We don't have any of that xref type, raise an exception
            raise UnknownXrefTypeException(
                "Unknown external reference type" " - {:s}.".format(xref_type)
            )
        return None

    def _resolve_extra_xref(self, xref_id):
        # Resolve any "extra" xrefs.
        def get_all_xrefs():
            r = self._extra_xrefs.loc[xref_id]
            return [r["oma"]] if len(r) == 1 else set(r["oma"])

        try:
            yield from map(
                lambda i: self.resolve_xref(i, _extra=False), get_all_xrefs()
            )
        except KeyError:
            pass

        x = self.resolve_xref(xref_id, _extra=False)
        if x is not None:
            yield x

    def get_xref_types(self):
        """
        Returns a list of the xref types.
        """
        return list(self._xrefs.keys())

    def get_xref(self, id, xref_type=None, sp=None):
        """
        Gets a specific type of external reference for a particular ID, if
        it exists.
        """
        sp = self.get_species(id) if sp is None else sp
        xref_type = "protId" if xref_type is None else xref_type

        if not self.is_oma or sp.NCBITaxID == -1:
            return sp[id].get(xref_type, None)
        else:
            # OMA *browser release* only. Therefore, must be protId.
            return "{:s}{:05d}".format(sp.species_code, id - sp.min_id + 1)

    def load_extra_xrefs(self, id_mapping):
        """
        Load the extra XRefs - these are not "typed" by different xref
        sources, so ensure to handle them carefully. The format is the same
        as the OMA download files.

        Note: there is currently an assertion that the is_oma flag must also
              be set, however this isn't strictly necessary.
        """
        if id_mapping is not None:
            assert self.is_oma is True, (
                "Safety check broken: extra xrefs " "may not work in the non-OMA case."
            )
            id_mapping = [id_mapping] if type(id_mapping) is str else id_mapping
            id_mapping = tqdm(
                id_mapping, desc="Loading XRef Mapping Files", unit=" files"
            )
            self._extra_xrefs = pd.concat(
                pd.read_csv(fn, sep="\t", comment="#", names=["oma", "xref"], dtype=str)
                for fn in id_mapping
            )
            self._extra_xrefs.set_index("xref", inplace=True)


class Species(object):
    """
    A class representing a species gene set from the OrthoXML file.
    """

    def __init__(self, el, no_xrefs=False, is_oma=False):
        """
        Initialises, given an element.
        """
        self.NCBITaxID = int(el.attrib["NCBITaxId"])
        self.name = el.attrib["name"]
        self.db_name = self._load_db_name(el)

        # Load the gene nodes.
        self.species_code = None
        self.min_id = None
        self.max_id = None
        self.genes = {}
        self._load_genes(el, no_xrefs, is_oma)

    def id_range(self):
        """
        Returns ID range that belongs to this species.
        """
        return range(self.min_id, self.max_id + 1)

    def __getitem__(self, id):
        """
        Returns a gene, so that we can reference the species like a dict of
        genes.
        """
        return self.genes[int(id)]

    def _load_db_name(self, el):
        """
        Loads the DB name up if we have it.
        """
        db = el.find("OrthoXML:database", NS)
        return db.attrib["name"] if db is not None else None

    def _load_genes(self, el, no_xrefs, is_oma):
        """
        Loads the genes up into the structure.
        """
        if not is_oma or self.NCBITaxID == -1:
            for gene in el.iterfind(".//OrthoXML:gene", NS):
                id_ = int(gene.attrib.pop("id", None))

                if id_ is not None:
                    if not no_xrefs:
                        # Backup - don't need these if OMA.
                        self.genes[id_] = copy(gene.attrib)

                    # Update the ID range
                    if self.min_id is not None:
                        if id_ < self.min_id:
                            self.min_id = id_
                        elif id_ > self.max_id:
                            self.max_id = id_
                    else:
                        self.min_id = self.max_id = id_
                else:
                    raise GeneIDException("No Gene ID found.")
        elif self.NCBITaxID != -1:
            # HOGs are from OMA -- i.e., a browser release.
            # Minimum ID is the first gene id
            genes = el.iterfind(".//OrthoXML:gene", NS)
            first_gene = next(genes)
            for gene in genes:
                pass
            last_gene = gene

            self.min_id = int(first_gene.attrib.pop("id"))
            self.max_id = int(last_gene.attrib.pop("id"))

            # Also get set the species code.
            self.species_code = first_gene.attrib.pop("protId")[:5]


class OrthoxmlParser(object):
    """
    HOG parser interface.
    """

    def __init__(
        self,
        fn,
        no_xrefs=False,
        load_species=True,
        cache_species=False,
        id_mapping=None,
        **kwargs
    ):
        self._load_if_oma(**kwargs)
        # Open context for OrthoXML
        self._set_context(fn)

        if load_species:
            p_fn = fn + ".species.p"
            if cache_species and os.path.isfile(p_fn):
                with open(p_fn, "rb") as fp:
                    self.species = pickle.load(fp)

            else:
                self.species = AllSpecies(self, no_xrefs, self.is_oma)
                if cache_species:
                    with open(p_fn, "wb") as fp:
                        pickle.dump(self.species, fp)

            # Load any extra xrefs
            self.species.load_extra_xrefs(id_mapping)

    def _load_if_oma(self, **kwargs):
        self.is_oma = kwargs.get("is_oma", False)
        is_not_oma = kwargs.get("is_not_oma", None)
        self.is_oma = False if is_not_oma and self.is_oma is True else self.is_oma

    def _set_context(self, fn=None, events=("start", "end"), tag=None):
        """
        Sets up the context.
        """
        if fn is not None:
            self.fp = auto_open(fn, "rb")
        else:
            self.fp.seek(0)

        # If "tag" is passed, set the parser context to only yield
        # tags of interest. This makes it parse faster since the barrier
        # native code -> python is not passed for the tags we're
        # not interested in
        if not tag:
            self.context = etree.iterparse(self.fp, events=events)
        else:
            self.context = etree.iterparse(self.fp, events=events, tag=tag)

    def reset(self, events=("start", "end"), tag=None):
        """
        Resets the context of the lxml parser. This is used
        to reinitialize the parser to parse different types of tags.
        """
        self._set_context(events=events, tag=tag)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.fp.close()
        del self.context

    def iter_species(self, no_xrefs=False, is_oma=False):
        """
        Iterates over species, yielding them until we hit the groups node.
        """

        # Set the parser to only yield species and groups, any namespace
        self.reset(tag=("{*}species", "{*}groups"))

        with tqdm(
            desc="Loading species",
            unit=" species",
            miniters=0,
            mininterval=1,
            maxinterval=60,
        ) as pbar:
            for ev, el in self.context:
                if ev == "end" and is_tag_name(el, "species"):
                    pbar.update(1)
                    yield Species(el, no_xrefs, is_oma)

                    el.clear()
                    while el.getprevious() is not None:
                        del el.getparent()[0]

                elif ev == "start" and is_tag_name(el, "groups"):
                    # Done with species
                    break

    def iter_hogs(self, auto_clean=False):
        """
        Iterates over HOGs, yielding them. If in low mem, remove them
        (call cleanup) as soon as done with them to save memory!

        auto_clean=True will delete the HOG from the XML structure after
        the yield.
        """

        self.reset(events=["end"], tag=("{*}species", "{*}orthologGroup", "{*}paralogGroup"))

        for ev, el in self.context:

            # If we don't clear the species, lxml will accumulate
            # their objects even if the context is not subscribed to them. Okay...
            if el.tag == tag_name("species"):
                el.clear()
                while el.getprevious() is not None:
                    del el.getparent()[0]

            if is_evolutionary_node(el) and el.getparent().tag == tag_name("groups"):
                hog = self._HOG(el)
                yield hog

                if auto_clean:
                    hog.cleanup()

    def _HOG(self, g):
        """
        Returns a HOG object for a node. This allows us to override this for
        HOGPROP and other uses.
        """
        return HOG(g)


HOGID_RE = re.compile(
    r"(?P<id>HOG:(?P<rel>[A-Z]+)?(?P<fam>\d+)(?:\.(?P<sub>[a-z0-9.]+))?)(?:_(?P<taxid>\d+))?"
)


class HOG(object):
    """
    Represents a single HOG XML node / tree.
    """

    def __init__(self, el):
        """
        Initialise the HOG object. This extracts the ID and saves the
        XML element.
        """
        self.struct = etree.fromstring(el) if isinstance(el, (str, bytes)) else el
        id_attr = self.struct.attrib.get("id")
        m = HOGID_RE.match(self.struct.attrib.get("id"))
        if m:
            self.fam = int(m.group("fam"))
            self.id = m.group("id")
            # self._loft_annotate_from_existing_xml_data(self.struct)
            self._loft_annotate()
        else:
            self.fam = self.id = int(self.struct.attrib.get("id"))
            self._loft_annotate()

    def _loft_annotate_from_existing_xml_data(self, node):
        if is_evolutionary_node(node):
            m = HOGID_RE.match(self.struct.attrib.get("id"))
            node.set("og", m.group("id"))
            for child in node:
                self._loft_annotate_from_existing_xml_data(child)

    def _loft_annotate(self):
        self._dup_counts = []
        self._loft_annotate_inner(self.struct, str(self.id))
        del self._dup_counts

    def _loft_annotate_inner(self, node, og, idx=0):
        if is_geneRef_node(node) or is_orthologGroup_node(node):
            node.set("og", og)
            if is_orthologGroup_node(node):
                for child in node:
                    self._loft_annotate_inner(child, og, idx)

        elif is_paralogGroup_node(node):
            idx += 1
            # Get next number at a given depth of duplication (idx)
            while len(self._dup_counts) < idx:
                self._dup_counts.append(0)
            self._dup_counts[idx - 1] += 1
            next_og = "{}.{}".format(og, self._dup_counts[idx - 1])
            node.set("og", next_og)
            for i, c in enumerate(node):
                letters = []
                nr = i
                while nr // 26 > 0:
                    letters.append(chr(97 + (nr % 26)))
                    nr = nr // 26 - 1
                letters.append(chr(97 + (nr % 26)))
                self._loft_annotate_inner(
                    c, "{}{}".format(next_og, "".join(letters[::-1])), idx
                )

    def geneRefs(self, node=None):
        """
        Iterate through the geneRef nodes in the HOG.
        """
        node = self.struct if node is None else node
        for gene in node.iterfind(".//OrthoXML:geneRef", NS):
            yield gene

    @lazy_property
    def gene_ids(self):
        """
        Returns a set of gene IDs which are in this HOG.
        """
        return self._gene_ids(None)

    def _gene_ids(self, node):
        return {int(g.attrib["id"]) for g in self.geneRefs(node)}

    def get_gene(self, i):
        """
        Gets the geneRef node i
        """
        return self.struct.xpath(
            r'.//OrthoXML:geneRef[re:test(@id, "{}")]'.format(str(i)), namespaces=NS
        )[0]

    def get_properties(self, node):
        """
        Finds and retrieves all direct children 'property' tags.
        """
        if node is not None:
            properties = node.findall("OrthoXML:property", NS)
        else:
            properties = self.struct.findall(".//OrthoXML:property", NS)

        if properties is not None:
            if node is not None:
                return {p.attrib["name"]: p.attrib["value"] for p in properties}
            else:
                return properties
        else:
            # No properties
            return None

    @lru_cache(None)
    def get_distance(self, node):
        """
        Retrieve the distance to the parent from a node, if it has been
        added into the XML.
        """
        properties = self.get_properties(node)
        if (
            properties is not None
            and "distance" in properties
            and properties["distance"] != "None"
        ):
            return float(properties["distance"])
        else:
            return None

    @lazy_property
    def levels(self):
        """
        Get the TaxRange levels of this HOG.
        """
        return {
            prop.attrib["value"]: prop.getparent()
            for prop in self.get_properties(None)
            if "name" in prop.attrib and prop.attrib["name"] == "TaxRange"
        }

    @lru_cache(None)
    def annotations(self, level=None):
        node = self.levels[level] if level is not None else self.struct
        return [
            y
            for x in node.findall(tag_name("notes"))
            for y in x.findall(tag_name("annotation"))
        ]

    def tostring(self, gz=False):
        """
        Serialises the XML that this object represents into a string and
        returns it.
        """
        s = etree.tostring(self.struct)
        return (
            s.decode("utf-8")
            if not gz
            else codecs.encode(codecs.encode(s, "zlib"), "base64").decode("utf-8")
        )

    def tonewick(self, _node=None, **kwargs):
        """
        Converts HOG to Newick format and returns it.

        Func will be called to return the distances on a node. This can be
        used to include belief in a certain label into the newick string...
        """
        toplevel = _node is None
        node = _node if _node is not None else self.struct

        if is_geneRef_node(node) or is_evolutionary_node(node):
            # Format this node's name / distance
            if is_geneRef_node(node):
                name = node.get("id", "")
                if kwargs.get("name_func"):
                    name = kwargs.get("name_func")(int(name))
            elif ("no_taxrange" not in kwargs) and (
                "TaxRange" in self.get_properties(node)
            ):
                name = self.get_properties(node)["TaxRange"]
            else:
                name = ""

            dist = (
                self.get_distance(node)
                if not kwargs.get("distance_func")
                else kwargs.get("distance_func")(self, node)
            )
            name += ":{:f}".format(dist) if dist is not None else ""

            children = list(
                filter(
                    lambda c: c is not None,
                    map(lambda n: self.tonewick(n, **kwargs), node),
                )
            )

            newick = (
                name if not children else "({:s}){:s}".format(", ".join(children), name)
            )

            return newick if not toplevel else "{:s};".format(newick)

    def tonhx(self, _node=None):
        """
        Converts HOG to NHX format and returns it.
        """
        return self.tonewick(_node=_node, nhx=True)

    def cleanup(self):
        """
        Cleanup the HOG, so that it will be removed by the GC.
        """
        self.struct.clear()
        for a in self.struct.xpath("ancestor-or-self::*"):
            while a.getprevious() is not None:
                del a.getparent()[0]
        del self.struct

    def __getstate__(self):
        """
        Returns the state of the object for pickle (multiproc uses this).
        """
        return (self.id, self.tostring())

    def __setstate__(self, state):
        """
        Loads the state of the object from a pickle (multiproc uses this).
        """
        (id, el_string) = state

        self.id = id
        self.struct = etree.fromstring(el_string)
