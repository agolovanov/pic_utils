"""Standalone on-axis Ez diagnostic for boosted-frame FBPIC simulations."""

import os

import h5py
import numpy as np
from scipy.constants import c


class BackTransformedOnAxisEzDiagnostic(object):
    """Write lab-frame on-axis Ez from a boosted-frame FBPIC simulation.

    The output is a single HDF5 file with ``Ez.shape == (Nt, Nz)``.  The
    diagnostic samples the same lab-frame slice trajectory as FBPIC's
    ``BackTransformedFieldDiagnostic``, but it stores only ``Ez`` at the first
    radial cell of the m=0 mode.
    """

    def __init__(
        self,
        zmin_lab,
        zmax_lab,
        v_lab,
        dt_snapshots_lab,
        Ntot_snapshots_lab,
        gamma_boost,
        fldobject,
        comm=None,
        write_dir='lab_diags_on_axis',
        period=100,
        filename='on_axis_Ez.h5',
    ):
        """Parameters
        ----------
        zmin_lab, zmax_lab : float
            Left and right edges of the lab-frame diagnostic window at t=0.

        v_lab : float
            Lab-frame velocity of the diagnostic window.

        dt_snapshots_lab : float
            Lab-frame time spacing between output rows.

        Ntot_snapshots_lab : int
            Number of lab-frame output rows.

        gamma_boost : float
            Lorentz factor of the boosted-frame simulation.

        fldobject : fbpic.fields.Fields
            The FBPIC field object, usually ``sim.fld``.

        comm : fbpic BoundaryCommunicator or None
            Usually ``sim.comm``.

        write_dir : str
            Directory in which the single HDF5 file is written.

        period : int
            Boosted-frame iteration interval for flushing buffered values.
            The diagnostic is still sampled every boosted-frame timestep.

        filename : str
            Name of the HDF5 file inside ``write_dir``.
        """
        self.fld = fldobject
        self.comm = comm
        self.rank = comm.rank if comm is not None else 0
        self.period = period

        self.gamma_boost = gamma_boost
        self.inv_gamma_boost = 1.0 / gamma_boost
        self.beta_boost = np.sqrt(1.0 - self.inv_gamma_boost**2)
        self.inv_beta_boost = 1.0 / self.beta_boost

        self.zmin_lab = zmin_lab
        self.zmax_lab = zmax_lab
        self.v_lab = v_lab
        self.dt_snapshots_lab = dt_snapshots_lab
        self.Nt = int(Ntot_snapshots_lab)

        self.dz_lab = c * self.fld.dt * self.inv_beta_boost * self.inv_gamma_boost
        self.Nz = int((zmax_lab - zmin_lab) / self.dz_lab) + 1
        self.inv_dz_lab = 1.0 / self.dz_lab

        self.write_dir = os.path.abspath(write_dir)
        self.filename = os.path.join(self.write_dir, filename)
        self.buffer = []

        if self.rank == 0:
            if not os.path.exists(self.write_dir):
                os.makedirs(self.write_dir)
            self._initialize_file()

    def _initialize_file(self):
        t = np.arange(self.Nt) * self.dt_snapshots_lab
        z = self.zmin_lab + self.v_lab * t[:, np.newaxis] + self.dz_lab * np.arange(self.Nz)[np.newaxis, :]

        with h5py.File(self.filename, 'w') as h5_file:
            dset = h5_file.create_dataset('Ez', shape=(self.Nt, self.Nz), dtype='f8', fillvalue=0.0)
            h5_file.create_dataset('t', data=t)
            h5_file.create_dataset('z', data=z)

            dset.attrs['units'] = 'V/m'
            h5_file['t'].attrs['units'] = 's'
            h5_file['z'].attrs['units'] = 'm'
            h5_file.attrs['description'] = 'Back-transformed lab-frame on-axis Ez'
            h5_file.attrs['zmin_lab'] = self.zmin_lab
            h5_file.attrs['zmax_lab'] = self.zmax_lab
            h5_file.attrs['v_lab'] = self.v_lab
            h5_file.attrs['dt_snapshots_lab'] = self.dt_snapshots_lab
            h5_file.attrs['gamma_boost'] = self.gamma_boost
            h5_file.attrs['dz_lab'] = self.dz_lab

    def write(self, iteration):
        """FBPIC diagnostic hook.

        FBPIC calls this every boosted-frame iteration.  Do not wrap this class
        in another period check, since each boosted-frame iteration can
        contribute a different lab-frame z slice.
        """
        records = self._local_records(iteration)

        if self.comm is not None and self.comm.size > 1:
            gathered_records = self.comm.mpi_comm.gather(records, root=0)
            if self.rank == 0:
                for proc_records in gathered_records:
                    self.buffer.extend(proc_records)
        elif self.rank == 0:
            self.buffer.extend(records)

        if self.rank == 0 and iteration % self.period == 0:
            self.flush()

    def _local_records(self, iteration):
        t_boost = iteration * self.fld.dt

        if self.comm is None:
            zmin_boost = self.fld.interp[0].zmin
            zmax_boost = self.fld.interp[0].zmax
            iz_offset = 0
        else:
            zmin_boost, zmax_boost = self.comm.get_zmin_zmax(
                local=True, with_damp=False, with_guard=False, rank=self.rank
            )
            iz_offset = self.comm.n_guard
            if self.comm.left_proc is None:
                iz_offset += self.comm.nz_damp + self.comm.n_inject

        dz_boost = self.fld.interp[0].dz
        Ez = self.fld.interp[0].Ez
        records = []

        for it_lab in range(self.Nt):
            t_lab = it_lab * self.dt_snapshots_lab
            zmin_snapshot = self.zmin_lab + self.v_lab * t_lab
            zmax_snapshot = self.zmax_lab + self.v_lab * t_lab

            z_boost = (t_lab * self.inv_gamma_boost - t_boost) * c * self.inv_beta_boost
            z_lab = (t_lab - t_boost * self.inv_gamma_boost) * c * self.inv_beta_boost

            if not (zmin_boost < z_boost < zmax_boost):
                continue
            if not (zmin_snapshot < z_lab < zmax_snapshot):
                continue

            iz_lab = int((z_lab - zmin_snapshot) * self.inv_dz_lab)
            if iz_lab < 0 or iz_lab >= self.Nz:
                continue

            z_gridunits = (z_boost - zmin_boost - 0.5 * dz_boost) / dz_boost
            iz_boost = int(z_gridunits)
            Sz = iz_boost + 1 - z_gridunits
            iz_boost += iz_offset

            ez_axis = Sz * Ez[iz_boost, 0] + (1.0 - Sz) * Ez[iz_boost + 1, 0]
            if hasattr(ez_axis, 'get'):
                ez_axis = ez_axis.get()

            records.append((it_lab, iz_lab, float(np.real(ez_axis))))

        return records

    def flush(self):
        """Write buffered samples to disk on rank 0."""
        if self.rank != 0 or len(self.buffer) == 0:
            return

        with h5py.File(self.filename, 'a') as h5_file:
            Ez = h5_file['Ez']
            for it_lab, iz_lab, ez_axis in self.buffer:
                Ez[it_lab, iz_lab] = ez_axis

        self.buffer = []

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass
