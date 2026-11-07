"""
Cell lists must be correct for triclinic boxes.

`set_up_cells` used to bin raw Cartesian coordinates into an orthogonal
|a| x |b| x |c| grid with a single +-L wrap, whatever the box shape. In a
tilted box this puts atoms in the wrong cells, so neighbours across the
tilted faces were silently lost, and an atom near the far corner of a box
with xy + xz > |a| got a cell index past the end of the cell vector, which
segfaulted on LAMMPS dumps of a 60 degree primitive cell.

The structure here is a 7x7x7 primitive fcc cell (343 atoms, so the cell
lists are used), written as a LAMMPS triclinic dump with a slight shear
(xy = xz = 0.51 |a|) and one atom just inside the far corner, as LAMMPS
writes it after the corner atom vibrates across the origin.
"""
import numpy as np
import pytest
import pyscal3.core as pc

NREP = 7
DNN = 2.5  # nearest-neighbour distance, second shell at ~3.5
TILT = 0.51


def _box():
    lx = NREP * DNN
    xy = xz = TILT * lx
    ly = np.sqrt(lx**2 - xy**2)
    yz = (lx**2 / 2 - xy * xz) / ly
    lz = np.sqrt(lx**2 - xz**2 - yz**2)
    return lx, ly, lz, xy, xz, yz


def _write_dump(path, corner_atom=True):
    lx, ly, lz, xy, xz, yz = _box()
    cell = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    g = np.arange(NREP) / NREP
    frac = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    if corner_atom:
        # the atom on the origin site, displaced slightly to -x, -y, -z and
        # wrapped back: it now sits just inside the far corner of the box
        frac[0] = 1.0 - 1e-3
    pos = frac @ cell
    xlo_b = min(0.0, xy, xz, xy + xz)
    xhi_b = lx + max(0.0, xy, xz, xy + xz)
    ylo_b = min(0.0, yz)
    yhi_b = ly + max(0.0, yz)
    with open(path, "w") as f:
        f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n%d\n" % len(pos))
        f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        f.write("%.16e %.16e %.16e\n" % (xlo_b, xhi_b, xy))
        f.write("%.16e %.16e %.16e\n" % (ylo_b, yhi_b, xz))
        f.write("%.16e %.16e %.16e\n" % (0.0, lz, yz))
        f.write("ITEM: ATOMS id type x y z\n")
        for i, p in enumerate(pos):
            f.write("%d 1 %.10f %.10f %.10f\n" % (i + 1, p[0], p[1], p[2]))
    return pos


def _counts(sys):
    return np.array([len(x) for x in sys.atoms.neighbors.index])


@pytest.mark.parametrize("corner_atom", [False, True])
def test_triclinic_dump_cutoff_cells(tmp_path, corner_atom):
    path = str(tmp_path / "tri.dump")
    pos = _write_dump(path, corner_atom=corner_atom)
    sys = pc.System(path, format="lammps-dump")
    assert sys.triclinic == 1
    assert sys.natoms > 250
    if corner_atom:
        # the condition that made the old binning index past the cell vector
        assert pos[:, 0].max() > 2 * sys.boxdims[0]

    sys.find.neighbors(method="cutoff", cutoff=3.0, cells=True)
    cells = _counts(sys)
    sys.find.neighbors(method="cutoff", cutoff=3.0, cells=False)
    brute = _counts(sys)
    assert np.all(brute == 12)
    assert np.array_equal(cells, brute)


def test_triclinic_dump_adaptive(tmp_path):
    path = str(tmp_path / "tri.dump")
    _write_dump(path, corner_atom=True)
    sys = pc.System(path, format="lammps-dump")
    sys.find.neighbors(method="cutoff", cutoff=0)
    assert np.all(_counts(sys) == 12)


def test_triclinic_dump_number(tmp_path):
    path = str(tmp_path / "tri.dump")
    _write_dump(path, corner_atom=True)
    sys = pc.System(path, format="lammps-dump")
    sys.find.neighbors(method="number", nmax=12)
    d = np.array([max(x) for x in sys.atoms.neighbors.distance])
    assert np.all(d < 3.0)
