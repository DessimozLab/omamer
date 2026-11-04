"""
    OMAmer - tree-driven and alignment-free protein assignment to sub-families

    This file is part of OMAmer.

    OMAmer is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OMAmer is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with OMAmer. If not, see <http://www.gnu.org/licenses/>.
"""
from collections import Counter

MAX_REPORTED = 10


class SpeciesTreeError(ValueError):
    pass


def load_species_tree(species_tree):
    """Return species_tree if it is a SpeciesTree, else read it as newick."""
    if isinstance(species_tree, SpeciesTree):
        return species_tree
    return SpeciesTree.read(species_tree)


class SpeciesTree(object):
    """
    Topology and taxon names of a newick species tree.

    The nodes are stored in preorder. Every node has a name (None if the
    newick has no label for it), a parent (index, -1 for the root), its
    children in the order of the newick file, and a level: the sum of the
    branch lengths to the root of the whole tree, where a missing branch
    length counts as 1 (i.e. the depth if there are no branch lengths).
    """

    def __init__(self, names, parents, children, levels, root=0):
        self._names = names
        self._parents = parents
        self._children = children
        self._levels = levels
        self._root = root

    @classmethod
    def from_string(cls, newick):
        import io

        return cls.read(io.StringIO(newick))

    @classmethod
    def read(cls, path):
        """
        Read a species tree in newick format, with labelled internal nodes,
        from a path or a file handle.
        """
        from Bio import Phylo

        # By default, Bio.Phylo stores internal node labels that look like
        # numbers (e.g. NCBI taxon ids) as confidence values and drops the
        # label. comments_are_confidence=True switches this off; the
        # confidence values are then taken from [...] comments instead,
        # which are not used here.
        tree = Phylo.read(path, "newick", comments_are_confidence=True)

        names, parents, children, levels = [], [], [], []
        stack = [(tree.root, -1, 0.0)]
        while stack:
            clade, parent, level = stack.pop()
            node = len(names)
            names.append(clade.name if clade.name else None)
            parents.append(parent)
            children.append([])
            levels.append(level)
            if parent >= 0:
                children[parent].append(node)
            # reversed, so that the children are visited in file order
            for child in reversed(clade.clades):
                length = child.branch_length if child.branch_length is not None else 1.0
                stack.append((child, node, level + length))
        return cls(names, parents, children, levels)

    @classmethod
    def from_orthoxml(cls, path):
        """
        Read the species tree from the <taxonomy> of an OrthoXML file
        (version 0.5 and later, where it is optional). Leaves referred to by
        a <species taxonId="..."> are named after the species. Only the start
        of the file is read, up to the end of the taxonomy or the groups.

        Raises SpeciesTreeError if the file has no taxonomy.
        """
        from lxml import etree

        def local(tag):
            return etree.QName(tag).localname if isinstance(tag, str) else None

        names, parents, children, levels = [], [], [], []
        taxon_ids = []
        species_of_taxon = {}
        stack = []
        found = False
        for event, el in etree.iterparse(str(path), events=("start", "end")):
            tag = local(el.tag)
            if event == "start":
                if tag == "taxonomy":
                    found = True
                elif tag == "taxon":
                    node = len(names)
                    parent = stack[-1] if stack else -1
                    names.append(el.get("name") or None)
                    parents.append(parent)
                    children.append([])
                    levels.append(levels[parent] + 1.0 if parent >= 0 else 0.0)
                    taxon_ids.append(el.get("id"))
                    if parent >= 0:
                        children[parent].append(node)
                    stack.append(node)
                elif tag in ("groups", "scores"):
                    break
            else:
                if tag == "taxon":
                    stack.pop()
                elif tag == "species":
                    if el.get("taxonId") is not None:
                        species_of_taxon[el.get("taxonId")] = el.get("name")
                    el.clear()
                elif tag == "taxonomy":
                    break
        if not found or not names:
            raise SpeciesTreeError("{} has no <taxonomy>".format(path))

        for node, taxon_id in enumerate(taxon_ids):
            if not children[node] and taxon_id in species_of_taxon:
                names[node] = species_of_taxon[taxon_id]
        return cls(names, parents, children, levels)

    @property
    def root(self):
        return self._names[self._root]

    def _nodes(self, node=None):
        """Node indices of the subtree of node, in preorder."""
        stack = [self._root if node is None else node]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(self._children[node]))

    def _find(self, name, what="taxon"):
        found = [n for n in self._nodes() if self._names[n] == name]
        if not found:
            raise SpeciesTreeError(
                "Unable to find {} as {} in the species tree".format(name, what)
            )
        if len(found) > 1:
            raise SpeciesTreeError(
                "Ambiguous {} {}: the species tree has {} nodes with this name".format(
                    what, name, len(found)
                )
            )
        return found[0]

    def __contains__(self, name):
        return any(self._names[n] == name for n in self._nodes())

    def check_taxon(self, name, what="taxon"):
        """Raise SpeciesTreeError if name is not exactly one node of the tree."""
        self._find(name, what)

    def subtree(self, name, what="root taxon"):
        """
        Return the subtree of the taxon. Its taxon names must be unique and
        no node may be unnamed, otherwise SpeciesTreeError is raised.
        """
        subtree = SpeciesTree(
            self._names, self._parents, self._children, self._levels,
            root=self._find(name, what),
        )
        subtree._check_names()
        return subtree

    def _check_names(self):
        names = [self._names[n] for n in self._nodes()]
        unnamed = names.count(None)
        if unnamed:
            raise SpeciesTreeError(
                "{} nodes of the species tree below {} have no name; all taxa "
                "must be named".format(unnamed, self.root)
            )
        duplicates = sorted(name for name, n in Counter(names).items() if n > 1)
        if duplicates:
            raise SpeciesTreeError(
                "taxon names of the species tree below {} must be unique, "
                "{} names are used more than once: {}{}".format(
                    self.root,
                    len(duplicates),
                    ", ".join(duplicates[:MAX_REPORTED]),
                    ", ..." if len(duplicates) > MAX_REPORTED else "",
                )
            )

    def taxa(self):
        """
        Yield (name, parent name, children names, level, is species,
        number of species below) for each taxon, in preorder. The parent of
        the root is None.
        """
        nodes = list(self._nodes())
        nspecies = {}
        for node in reversed(nodes):
            children = self._children[node]
            nspecies[node] = sum(nspecies[c] for c in children) if children else 1
        for node in nodes:
            parent = self._parents[node] if node != self._root else -1
            yield (
                self._names[node],
                self._names[parent] if parent >= 0 else None,
                [self._names[c] for c in self._children[node]],
                self._levels[node],
                not self._children[node],
                nspecies[node],
            )

    def parents(self):
        """Taxon name -> name of its parent (None for the root)."""
        return {name: parent for name, parent, *_ in self.taxa()}
