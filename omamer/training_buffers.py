"""Persist the mkdb sequence buffers needed to refit family models.

An OMAmer database stores k-mer tables but not the sequences they came from,
because search never needs them. Beta-binomial fitting does: its null is the
set of indexed proteins, so it has to re-derive each protein's k-mer set.

Keeping the buffers in a sidecar rather than in the database itself is
deliberate. A refit sweep produces one database per (k-mer filter, N grid)
cell, and every cell would otherwise carry a redundant copy of buffers that
only the fit reads.

Record order is the database's own protein order, so a refit can recover the
family each null query belongs to and exclude it from that family's own
background -- which a FASTA-driven fit cannot do reliably.
"""

from __future__ import annotations

import numpy as np
import tables

from ._utils import LOG

MODALITIES = ("seq", "ss")


def _record_bounds(sequence_buffer):
    """Return start/stop offsets of every space-delimited record."""
    byte_view = np.asarray(sequence_buffer).view(np.uint8)
    stops = np.flatnonzero(byte_view == ord(" ")).astype(np.int64)
    starts = np.empty(stops.size, dtype=np.int64)
    if starts.size:
        starts[0] = 0
        starts[1:] = stops[:-1] + 1
    return starts, stops


def write_training_buffers(path, k, alphabet_n, n_records, buffers):
    """Write the sequence buffers a later refit needs.

    ``buffers`` maps a modality name to the buffer mkdb assembled, or to None
    for a modality this build does not have.
    """
    filters = tables.Filters(complevel=6, complib="blosc")
    with tables.open_file(str(path), mode="w") as h5:
        h5.root._v_attrs["k"] = int(k)
        h5.root._v_attrs["alphabet_n"] = int(alphabet_n)
        h5.root._v_attrs["n_records"] = int(n_records)
        stored = []
        for modality in MODALITIES:
            buffer = buffers.get(modality)
            if buffer is None:
                continue
            byte_view = np.asarray(buffer).view(np.uint8)
            starts, stops = _record_bounds(byte_view)
            if starts.size != int(n_records):
                raise ValueError(
                    "{} buffer holds {} records, expected {}".format(
                        modality, starts.size, n_records
                    )
                )
            group = h5.create_group("/", modality)
            h5.create_carray(group, "Buffer", obj=byte_view, filters=filters)
            h5.create_carray(group, "Starts", obj=starts, filters=filters)
            h5.create_carray(group, "Stops", obj=stops, filters=filters)
            stored.append(modality)
            LOG.debug(
                " - stored %s training buffer, %.2f MB for %d records",
                modality,
                byte_view.nbytes / (1024 * 1024),
                starts.size,
            )
        h5.root._v_attrs["modalities"] = ",".join(stored)
    LOG.info("Wrote training buffers for %s to %s", ",".join(stored), path)
    return tuple(stored)


class TrainingBuffers:
    """Read-only accessor for a training-buffer sidecar."""

    def __init__(self, path):
        self.path = str(path)
        self.h5 = tables.open_file(self.path, mode="r")
        attrs = self.h5.root._v_attrs
        self.k = int(attrs["k"])
        self.alphabet_n = int(attrs["alphabet_n"])
        self.n_records = int(attrs["n_records"])
        self.modalities = tuple(
            m for m in str(attrs["modalities"]).split(",") if m
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def check_compatible(self, db):
        """Fail loudly when the sidecar does not belong to this database."""
        n_proteins = len(db.protein_table)
        if self.n_records != n_proteins:
            raise ValueError(
                "Training buffers hold {} records but the database has {} "
                "proteins -- the sidecar belongs to a different build".format(
                    self.n_records, n_proteins
                )
            )
        if self.k != int(db.ki.k):
            raise ValueError(
                "Training buffers were written for k={} but the database "
                "index uses k={}".format(self.k, db.ki.k)
            )
        if self.alphabet_n != int(db.ki.alphabet.n):
            raise ValueError(
                "Training buffers use alphabet size {} but the database "
                "index uses {}".format(self.alphabet_n, db.ki.alphabet.n)
            )

    def buffer(self, modality, in_memory=True):
        if modality not in self.modalities:
            raise ValueError(
                "Training buffers contain no '{}' modality (have: {})".format(
                    modality, ", ".join(self.modalities) or "none"
                )
            )
        node = self.h5.get_node("/{}/Buffer".format(modality))
        return node[:] if in_memory else node

    def bounds(self, modality):
        starts = self.h5.get_node("/{}/Starts".format(modality))[:]
        stops = self.h5.get_node("/{}/Stops".format(modality))[:]
        return starts, stops
