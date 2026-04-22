import tables
from pathlib import Path

def check_is_hdf5(filename):
    """
    Checks the given file is an existing HDF5 file.
    """
    p = Path(filename)
    if not p.exists():
        raise RuntimeError(f"{filename} does not exist")

    if p.suffix.lower() not in {".h5", ".hdf5"}:
        raise RuntimeError(f"{filename} is not an HDF5 file")


class H5Adapter:
    """
    Small wrapper around an opened HDF5 file.
    Can open both OmaServer.h5 and structure_db.h5
    """

    def __init__(self, filename):
        self.filename = filename
        self.handle = None
        self.oma_version = None

    def open(self):
        check_is_hdf5(self.filename)
        self.handle = tables.open_file(self.filename, mode="r")
        return self

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()


class H5IsNotOMAError(Exception):
    pass


def check_h5_is_oma(h5handle):
    """Checks the given h5 handle (opened h5 file) is an OMA DB."""
    try:
        return h5handle.get_node_attr("/", "oma_version")
    except AttributeError:
        raise H5IsNotOMAError


class OmaServerAdapter(H5Adapter):
    def __init__(self, filename):
        super().__init__(filename)

    def open(self):
        super().open()

        try:
            self.oma_version = check_h5_is_oma(self.handle)
        except Exception:
            self.close()
            raise

        return self


class OmaStructureDBAdapter(H5Adapter):
    """
    Wrapper for the OMA 3di structure database file.
    Takes ${OMA_RELEASE}/structure/structure_db.h5 file as input
    """
    def __init__(self, filename):
        super().__init__(filename)
