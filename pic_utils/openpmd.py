import numpy as _np
import pandas as _pd
import pint as _pint
from openpmd_viewer.addons import LpaDiagnostics as _Lpa

import typing

if typing.TYPE_CHECKING:
    from .plasma import PlasmaUnits


class OpenPMDWrapper:
    def __init__(self, folder, plasma_units: 'PlasmaUnits | None' = None) -> None:
        self.simulation = _Lpa(folder)
        self.ureg = _pint.get_application_registry()
        self.plasma_units = plasma_units

        self.c = self.ureg('speed_of_light')

    def _get_field_units(self, field: str):
        if field == 'E':
            return self.ureg('V/m')
        if field == 'B':
            return self.ureg('tesla')
        if field == 'J':
            return self.ureg('C/m^2/s')
        if field.startswith('rho'):
            return self.ureg('C/m^3')
        return self.ureg.dimensionless

    def read_field(
        self,
        iteration: int,
        field: str,
        component: str | None = None,
        *,
        geometry: typing.Literal['xz', 'yz', 'xy', '3d'] = 'xz',
        grid: bool = False,
        mode: int | typing.Literal['all'] = 'all',
        only_positive_r: bool = False,
        slice_relative_position: float | list[float] | None = None,
        plasma_units: 'PlasmaUnits | typing.Literal["auto"] | None' = 'auto',
    ) -> '_np.ndarray | _pint.Quantity | tuple[_np.ndarray | _pint.Quantity, ...]':
        """
        Read an openPMD field component in one of the supported geometries.

        The field is read through openPMD-viewer's ``get_field`` API and then
        converted to pint units, or to the configured plasma units when
        requested.

        Parameters
        ----------
        iteration : int
            Iteration number to read.
        field : str
            Field record name, for example ``'E'``, ``'B'``, ``'J'`` or a
            charge-density record such as ``'rho_electrons'``.
        component : str | None, optional
            Field component for vector records, for example ``'x'``, ``'y'``
            or ``'z'``. Use ``None`` for scalar records.
        geometry : {'xz', 'yz', 'xy', '3d'}, optional
            Geometry to return:

            - ``'xz'``: theta-mode plane at ``theta = 0``.
            - ``'yz'``: theta-mode plane at ``theta = pi / 2``.
            - ``'xy'``: Cartesian transverse slice at fixed ``z``.
            - ``'3d'``: full Cartesian reconstruction.

        grid : bool, optional
            When ``True``, return coordinate grids together with the field.
        mode : int | {'all'}, optional
            Azimuthal mode passed to openPMD-viewer. Use ``'all'`` to sum all
            available modes.
        only_positive_r : bool, optional
            For ``'xz'`` and ``'yz'``, keep only the non-negative radial half
            of the plane.
        slice_relative_position : float | list[float] | None, optional
            Relative slice position passed to openPMD-viewer for ``'xy'``
            geometry. Values are in ``[-1, 1]``; ``0`` is the center of the
            simulation box. This argument is only supported for ``'xy'``.
        plasma_units : PlasmaUnits | {'auto'} | None, optional
            Unit conversion policy. ``'auto'`` uses the wrapper's configured
            plasma units, ``None`` keeps pint quantities, and a ``PlasmaUnits``
            instance converts the field and grids to unitless plasma units.

        Returns
        -------
        numpy.ndarray | pint.Quantity | tuple
            If ``grid`` is ``False``, returns the field array.

            If ``grid`` is ``True``, returns:

            - ``(zz, xx, f)`` for ``'xz'`` and ``'yz'``.
            - ``(xx, yy, f)`` for ``'xy'``.
            - ``(zz, xx, yy, f)`` for ``'3d'``.

        Raises
        ------
        ValueError
            If ``geometry`` is unsupported, if ``only_positive_r`` is used
            with ``'xy'`` or ``'3d'``, or if ``slice_relative_position`` is
            provided for a geometry other than ``'xy'``.
        """
        if geometry != 'xy' and slice_relative_position is not None:
            raise ValueError(f'Value {slice_relative_position=} is only allowed when geometry="xy"')

        if geometry == 'xz':
            theta = 0.0
            slice_across = None
        elif geometry == 'yz':
            theta = _np.pi / 2
            slice_across = None
        elif geometry == 'xy':
            if only_positive_r:
                raise ValueError(f'Value {only_positive_r=} is not allowed when {geometry=}')
            theta = None
            slice_across = 'z'
        elif geometry == '3d':
            if only_positive_r:
                raise ValueError(f'Value {only_positive_r=} is not allowed when {geometry=}')
            theta = None
            slice_across = None
        else:
            raise ValueError(f'Geometry {geometry} is not available, only xz, yz, xy, 3d')

        if plasma_units == 'auto':
            plasma_units = self.plasma_units

        f, f_info = self.simulation.get_field(
            field,
            component,
            iteration=iteration,
            theta=theta,
            m=mode,
            slice_across=slice_across,
            slice_relative_position=slice_relative_position,
        )

        f = f * self._get_field_units(field)

        if plasma_units is not None:
            f = plasma_units.convert_to_unitless(f)

        if geometry == '3d':
            if grid:
                xx, yy, zz = _np.meshgrid(f_info.x, f_info.y, f_info.z, indexing='ij')
                zz = (zz * self.ureg.m).to('um')
                xx = (xx * self.ureg.m).to('um')
                yy = (yy * self.ureg.m).to('um')
                if plasma_units is not None:
                    xx = plasma_units.convert_to_unitless(xx)
                    yy = plasma_units.convert_to_unitless(yy)
                    zz = plasma_units.convert_to_unitless(zz)
                return zz, xx, yy, f
            else:
                return f
        elif geometry == 'xy':
            if grid:
                xx, yy = _np.meshgrid(f_info.x, f_info.y, indexing='ij')
                xx = (xx * self.ureg.m).to('um')
                yy = (yy * self.ureg.m).to('um')
                if plasma_units is not None:
                    xx = plasma_units.convert_to_unitless(xx)
                    yy = plasma_units.convert_to_unitless(yy)
                return xx, yy, f
            else:
                return f
        else:
            r = f_info.r
            if only_positive_r:
                index = len(r) // 2
                f = f[index:, :]
                r = r[index:]
            if grid:
                zz, xx = _np.meshgrid(f_info.z, r)
                zz = (zz * self.ureg.m).to('um')
                xx = (xx * self.ureg.m).to('um')
                if plasma_units is not None:
                    xx = plasma_units.convert_to_unitless(xx)
                    zz = plasma_units.convert_to_unitless(zz)
                return zz, xx, f
            else:
                return f

    def read_poynting_vector(
        self,
        iteration,
        component,
        *,
        geometry: typing.Literal['xz', 'yz', 'xy', '3d'] = 'xz',
        grid=False,
        mode='all',
        only_positive_r=False,
        slice_relative_position=None,
    ):
        from .electromagnetism import poynting_vector

        if component != 'z':
            raise NotImplementedError(f'Components other than z are not implemented yet, {component} not available')

        kwargs = {
            'geometry': geometry,
            'mode': mode,
            'only_positive_r': only_positive_r,
            'slice_relative_position': slice_relative_position,
        }

        ex = self.read_field(iteration, 'E', 'x', grid=grid, **kwargs)
        ey = self.read_field(iteration, 'E', 'y', grid=False, **kwargs)

        bx = self.read_field(iteration, 'B', 'x', grid=False, **kwargs)
        by = self.read_field(iteration, 'B', 'y', grid=False, **kwargs)

        if grid:
            if geometry == '3d':
                zz, xx, yy, ex = ex
                return zz, xx, yy, poynting_vector(ex, ey, bx, by)
            else:
                xx, yy, ex = ex
                return xx, yy, poynting_vector(ex, ey, bx, by)
        else:
            return poynting_vector(ex, ey, bx, by)

    def read_fluence(
        self,
        iteration,
        *,
        geometry: typing.Literal['x', 'y', 'xy'] = 'xy',
        grid=False,
        mode='all',
        only_positive_r=False,
    ):
        """
        Calculate the transverse fluence distribution from the longitudinal Poynting vector.

        The fluence is evaluated from a field snapshot as ``integral(S_z dz) / c``.

        Parameters
        ----------
        iteration : int
            Iteration number to read.
        geometry : {'x', 'y', 'xy'}, optional
            Transverse distribution to return. ``'x'`` and ``'y'`` return 1D
            profiles calculated from the corresponding longitudinal plane.
            ``'xy'`` returns a 2D transverse map calculated from the full 3D
            reconstruction.
        grid : bool, optional
            When ``True``, return transverse grid coordinates together with the
            fluence distribution.
        mode : int | {'all'}, optional
            Azimuthal mode passed to openPMD-viewer.
        only_positive_r : bool, optional
            For ``'x'`` and ``'y'``, keep only the non-negative radial half of
            the source longitudinal plane.

        Returns
        -------
        pint.Quantity | tuple
            Fluence in ``J/m^2``. With ``grid=True``, returns ``(x, fluence)``
            for ``'x'``, ``(y, fluence)`` for ``'y'``, and ``(xx, yy, fluence)``
            for ``'xy'``.
        """
        if geometry == 'x':
            source_geometry = 'xz'
        elif geometry == 'y':
            source_geometry = 'yz'
        elif geometry == 'xy':
            if only_positive_r:
                raise ValueError(f'Value {only_positive_r=} is not allowed when {geometry=}')
            source_geometry = '3d'
        else:
            raise ValueError(f'Geometry {geometry} is not available, only x, y, xy')

        data = self.read_poynting_vector(
            iteration,
            'z',
            geometry=source_geometry,
            grid=True,
            mode=mode,
            only_positive_r=only_positive_r,
        )

        if geometry == 'xy':
            zz, xx, yy, sz = data
            z = zz[0, 0, :].to('m')
            fluence = (_np.trapezoid(sz, z, axis=2) / self.c).to('J/m^2')
            if grid:
                return xx[:, :, 0].copy(), yy[:, :, 0].copy(), fluence
            else:
                return fluence
        else:
            zz, transverse, sz = data
            z = zz[0, :].to('m')
            fluence = (_np.trapezoid(sz, z, axis=1) / self.c).to('J/m^2')
            if grid:
                return transverse[:, 0].copy(), fluence
            else:
                return fluence

    def read_cylindrical_mode(
        self,
        iteration: int,
        field: str,
        component: str | None,
        mode: int,
        *,
        grid: bool = False,
        only_positive_r: bool = False,
    ):
        """
        Read the raw complex cylindrical amplitude for one theta mode.

        openPMD-viewer exposes theta-mode fields through reconstructed planes,
        but its theta-mode datasets store real coefficients: ``m=0`` at index
        0 and, for ``m > 0``, cosine/sine coefficients at indices
        ``2*m - 1`` and ``2*m``. This method returns the equivalent complex
        amplitude ``cos + 1j * sin`` on the ``(r, z)`` grid.
        """
        if not isinstance(mode, int) or isinstance(mode, bool) or mode < 0:
            raise ValueError(f'Value {mode=} is not allowed, use a non-negative integer mode')

        modes = self.available_cylindrical_modes(field)
        if mode not in modes:
            raise ValueError(f'Mode {mode} is not available. Available modes: {modes}')

        data, z, r = self._read_raw_cylindrical_mode_data(iteration, field, component, mode, only_positive_r)

        data = data * self._get_field_units(field)

        if grid:
            zz, rr = _np.meshgrid(z, r)
            return (zz * self.ureg.m).to('um'), (rr * self.ureg.m).to('um'), data
        else:
            return data

    def calculate_mode_power_density(
        self,
        iteration: int,
        mode: int | typing.Literal['all'] = 'all',
        *,
        density: typing.Literal['radial', 'angular_average'] = 'angular_average',
        grid: bool = False,
        only_positive_r: bool = False,
    ):
        """
        Calculate the angular integral of the longitudinal Poynting flux.

        Parameters
        ----------
        iteration : int
            Iteration number to read.
        mode : int | {'all'}, optional
            Azimuthal mode. ``'all'`` sums the independent contributions of
            all available modes.
        density : {'radial', 'angular_average'}, optional
            ``'radial'`` returns ``r * integral(S_z dtheta)`` in ``W/m``.
            ``'angular_average'`` returns ``integral(S_z dtheta) / (2*pi)``
            in ``W/m^2``.

            Defaults to ``'angular_average'``.
        grid : bool, optional
            When ``True``, return ``(z, r, density)`` coordinate grids.
        only_positive_r : bool, optional
            When ``True``, keep only the non-negative radial half of the
            theta-mode plane. By default, return the full signed radial axis.

        Returns
        -------
        pint.Quantity | tuple
            Power density array on the ``(r, z)`` grid.
        """
        if density not in ('radial', 'angular_average'):
            raise ValueError(f'Unknown {density=}; use "radial" or "angular_average"')

        if mode == 'all':
            modes = sorted(set(self.available_cylindrical_modes('E')) & set(self.available_cylindrical_modes('B')))
            if len(modes) == 0:
                raise ValueError('No common modes available for E and B fields')
            power_density = self.calculate_mode_power_density(
                iteration, modes[0], density=density, grid=grid, only_positive_r=only_positive_r
            )
            if grid:
                zz, rr, power_density = power_density
            for mode in modes[1:]:
                power_density += self.calculate_mode_power_density(
                    iteration, mode, density=density, grid=False, only_positive_r=only_positive_r
                )

            return power_density if not grid else (zz, rr, power_density)

        from .units import real, conj

        mu0 = self.ureg('vacuum_permeability')

        force_grid = grid or density == 'radial'

        er = self.read_cylindrical_mode(iteration, 'E', 'r', mode, grid=force_grid, only_positive_r=only_positive_r)
        if force_grid:
            zz, rr, er = er
        et = self.read_cylindrical_mode(iteration, 'E', 't', mode, only_positive_r=only_positive_r)
        br = self.read_cylindrical_mode(iteration, 'B', 'r', mode, only_positive_r=only_positive_r)
        bt = self.read_cylindrical_mode(iteration, 'B', 't', mode, only_positive_r=only_positive_r)

        if mode == 0:
            mode_integral = real(er * bt - et * br) / mu0
        else:
            mode_integral = 0.5 * real(er * conj(bt) - et * conj(br)) / mu0
        mode_integral = mode_integral.to('W/m^2')

        if density == 'angular_average':
            result = mode_integral
        else:
            result = 2 * _np.pi * (rr.to('m') * mode_integral).to('W/m')

        if grid:
            return zz, rr, result
        else:
            return result

    def calculate_mode_energy_density(
        self,
        iteration: int,
        mode: int | typing.Literal['all'] = 'all',
        *,
        method: typing.Literal['field', 'flux'] = 'field',
        density: typing.Literal['radial', 'angular_average'] = 'angular_average',
        grid: bool = False,
        only_positive_r: bool = False,
    ):
        """
        Calculate the local EM energy density on the cylindrical ``(r, z)`` grid.

        Parameters
        ----------
        iteration : int
            Iteration number to read.
        mode : int | {'all'}, optional
            Azimuthal mode. ``'all'`` sums the independent contributions of
            all available modes.
        method : {'field', 'flux'}, optional
            ``'field'`` calculates the true EM energy density from
            ``eps0 * E^2 / 2 + B^2 / (2 * mu0)``. ``'flux'`` calculates the
            laser-like estimate ``S_z / c`` from the longitudinal Poynting
            flux.

            Defaults to ``'field'``.
        density : {'radial', 'angular_average'}, optional
            ``'radial'`` returns ``r * integral(u dtheta)`` in ``J/m^2``.
            ``'angular_average'`` returns ``integral(u dtheta) / (2*pi)`` in
            ``J/m^3``.

            Defaults to ``'angular_average'``.
        grid : bool, optional
            When ``True``, return ``(z, r, density)`` coordinate grids.
        only_positive_r : bool, optional
            When ``True``, keep only the non-negative radial half of the
            theta-mode plane. By default, return the full signed radial axis.

        Returns
        -------
        pint.Quantity | tuple
            Energy density array on the ``(r, z)`` grid.
        """
        if method not in ('field', 'flux'):
            raise ValueError(f'Unknown {method=}; use "field" or "flux"')

        if density not in ('radial', 'angular_average'):
            raise ValueError(f'Unknown {density=}; use "radial" or "angular_average"')

        if method == 'flux':
            zz, rr, power_density = self.calculate_mode_power_density(
                iteration, mode, density=density, grid=True, only_positive_r=only_positive_r
            )
            units = 'J/m^3' if density == 'angular_average' else 'J/m^2'
            energy_density = (power_density / self.c).to(units)
            return energy_density if not grid else (zz, rr, energy_density)

        if mode == 'all':
            modes = sorted(set(self.available_cylindrical_modes('E')) & set(self.available_cylindrical_modes('B')))
            if len(modes) == 0:
                raise ValueError('No common modes available for E and B fields')
            energy_density = self.calculate_mode_energy_density(
                iteration, modes[0], method=method, density=density, grid=grid, only_positive_r=only_positive_r
            )
            if grid:
                zz, rr, energy_density = energy_density
            for mode in modes[1:]:
                energy_density += self.calculate_mode_energy_density(
                    iteration, mode, method=method, density=density, grid=False, only_positive_r=only_positive_r
                )

            return energy_density if not grid else (zz, rr, energy_density)

        from .units import conj, real

        mu0 = self.ureg('vacuum_permeability')
        eps0 = self.ureg('vacuum_permittivity')

        force_grid = grid or density == 'radial'

        er = self.read_cylindrical_mode(iteration, 'E', 'r', mode, grid=force_grid, only_positive_r=only_positive_r)
        if force_grid:
            zz, rr, er = er
        et = self.read_cylindrical_mode(iteration, 'E', 't', mode, only_positive_r=only_positive_r)
        ez = self.read_cylindrical_mode(iteration, 'E', 'z', mode, only_positive_r=only_positive_r)
        br = self.read_cylindrical_mode(iteration, 'B', 'r', mode, only_positive_r=only_positive_r)
        bt = self.read_cylindrical_mode(iteration, 'B', 't', mode, only_positive_r=only_positive_r)
        bz = self.read_cylindrical_mode(iteration, 'B', 'z', mode, only_positive_r=only_positive_r)

        if mode == 0:
            e2 = real(er * er + et * et + ez * ez)
            b2 = real(br * br + bt * bt + bz * bz)
        else:
            e2 = 0.5 * real(er * conj(er) + et * conj(et) + ez * conj(ez))
            b2 = 0.5 * real(br * conj(br) + bt * conj(bt) + bz * conj(bz))

        angular_average = (0.5 * (eps0 * e2 + b2 / mu0)).to('J/m^3')

        if density == 'angular_average':
            result = angular_average
        else:
            result = 2 * _np.pi * (rr.to('m') * angular_average).to('J/m^2')

        if grid:
            return zz, rr, result
        else:
            return result

    def calculate_mode_radial_energy_profile(
        self,
        iteration: int,
        mode: int | typing.Literal['all'] = 'all',
        *,
        method: typing.Literal['field', 'flux'] = 'field',
        density: typing.Literal['radial', 'angular_average'] = 'angular_average',
        grid: bool = False,
        only_positive_r: bool = False,
    ):
        """
        Calculate the radial EM energy profile integrated over ``z``.

        Parameters
        ----------
        iteration : int
            Iteration number to read.
        mode : int | {'all'}, optional
            Azimuthal mode. ``'all'`` sums the independent contributions of
            all available modes.
        method : {'field', 'flux'}, optional
            ``'field'`` integrates the true EM energy density. ``'flux'`` uses
            the laser-like Poynting estimate ``S_z / c``.

            Defaults to ``'field'``.
        density : {'radial', 'angular_average'}, optional
            ``'radial'`` returns ``integral(r * integral(u dtheta) dz)`` in
            ``J/m``. ``'angular_average'`` returns
            ``integral(integral(u dtheta) / (2*pi) dz)`` in ``J/m^2``.

            Defaults to ``'angular_average'``.
        grid : bool, optional
            When ``True``, return ``(r, profile)``.
        only_positive_r : bool, optional
            When ``True``, keep only the non-negative radial half of the
            theta-mode plane. By default, return the full signed radial axis.

        Returns
        -------
        pint.Quantity | tuple
            Energy profile on the radial grid.
        """
        zz, rr, energy_density = self.calculate_mode_energy_density(
            iteration, mode, method=method, density=density, grid=True, only_positive_r=only_positive_r
        )

        z = zz[0, :].to('m')
        units = 'J/m^2' if density == 'angular_average' else 'J/m'
        energy_profile = _np.trapezoid(energy_density, z, axis=1).to(units)

        if grid:
            return rr[:, 0].copy(), energy_profile
        else:
            return energy_profile

    def calculate_mode_power(self, iteration: int, mode: int | typing.Literal['all'] = 'all', *, grid: bool = False):
        """
        Calculate the total laser power in a cylindrical mode.

        This integrates the positive-r radial power density over radius.

        Parameters
        ----------
        iteration : int
            Iteration number to read.
        mode : int | {'all'}, optional
            Azimuthal mode. ``'all'`` sums the independent contributions of
            all available modes.
        grid : bool, optional
            When ``True``, return ``(z, power)``.

        Returns
        -------
        pint.Quantity | tuple
            Power profile in ``W`` on the longitudinal grid.
        """
        zz, rr, power_density = self.calculate_mode_power_density(
            iteration, mode, density='radial', grid=True, only_positive_r=True
        )
        power = _np.trapezoid(power_density, rr[:, 0].to('m'), axis=0).to('W')

        if grid:
            return zz[0, :].copy(), power
        else:
            return power

    def calculate_mode_energy(
        self,
        iteration: int,
        mode: int | typing.Literal['all'] = 'all',
        *,
        method: typing.Literal['field', 'flux'] = 'field',
    ):
        """
        Calculate the total laser energy in a cylindrical mode.

        This integrates the positive-r radial energy density over radius.

        Parameters
        ----------
        iteration : int
            Iteration number to read.
        mode : int | {'all'}, optional
            Azimuthal mode. ``'all'`` sums the independent contributions of
            all available modes.
        method : {'field', 'flux'}, optional
            ``'field'`` integrates the true EM energy density. ``'flux'`` uses
            the laser-like Poynting estimate ``S_z / c``.

            Defaults to ``'field'``.

        Returns
        -------
        pint.Quantity
            Total energy in ``J``.
        """
        r, energy_profile = self.calculate_mode_radial_energy_profile(
            iteration, mode, method=method, density='radial', grid=True, only_positive_r=True
        )
        return _np.trapezoid(energy_profile, r.to('m')).to('J')

    def available_cylindrical_modes(self, field: str) -> list[int]:
        if field not in self.simulation.avail_fields:
            raise ValueError(f'Field {field} is not available')
        metadata = self.simulation.fields_metadata[field]
        if metadata['geometry'] != 'thetaMode':
            raise ValueError(f'Field {field} has geometry {metadata["geometry"]}, not thetaMode')
        return [int(m) for m in metadata['avail_circ_modes'] if m != 'all']

    def _read_raw_cylindrical_mode_data(
        self,
        iteration: int,
        field: str,
        component: str | None,
        mode: int,
        only_positive_r: bool,
    ) -> 'tuple[_np.ndarray, _np.ndarray, _np.ndarray]':
        self.simulation._find_output(None, iteration)
        iteration = self.simulation.iterations[self.simulation._current_i]

        if mode == 0:
            data, info = self.simulation.data_reader.read_field_circ(iteration, field, component, None, None, mode, 0.0)
            data = data.astype(complex)
        else:
            data_cos, info = self.simulation.data_reader.read_field_circ(
                iteration, field, component, None, None, mode, 0.0
            )
            data_sin, _ = self.simulation.data_reader.read_field_circ(
                iteration, field, component, None, None, mode, _np.pi / (2 * mode)
            )
            data = data_cos + 1j * data_sin

        data, r = self._normalize_theta_mode_data(data, info, only_positive_r)
        return data, info.z, r

    def _normalize_theta_mode_data(self, data, info, only_positive_r):
        r_axis = next(axis for axis, label in info.axes.items() if label == 'r')
        if r_axis == 1:
            data = data.T

        if not only_positive_r:
            return data, info.r

        index = len(info.r) // 2
        return data[index:, :], info.r[index:]

    def read_particles(
        self,
        iteration,
        species,
        *,
        parameters=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'],
        select=None,
        initialize_energy=True,
        sort_by=None,
        max_particle_number=None,
    ):
        from .bunch import gamma, gamma_to_energy, limit_particle_number

        specie_data_arr = []

        if isinstance(species, str):
            species = [species]

        for i, s in enumerate(species):
            specie_data_list = self.simulation.get_particle(parameters, species=s, iteration=iteration, select=select)
            specie_data = _pd.DataFrame.from_dict(dict(zip(parameters, specie_data_list)))
            specie_data['species_id'] = i
            specie_data_arr.append(specie_data)

        if len(specie_data_arr) > 0:
            data = _pd.concat(specie_data_arr)
        else:
            dataframe_columns = parameters + ['species_id']
            data = _pd.DataFrame(columns=dataframe_columns)

        if initialize_energy:
            data['gamma'] = gamma(data['ux'], data['uy'], data['uz'])
            data['energy'] = gamma_to_energy(data['gamma']).m_as('MeV')

        if max_particle_number is not None:
            data = limit_particle_number(data, max_particle_number)

        if sort_by is not None:
            data.sort_values(sort_by, inplace=True)

        return data

    def iterations(self):
        return self.simulation.iterations

    def times(self, units='s'):
        return self.simulation.t * self.ureg(units)

    def species(self):
        species = self.simulation.avail_species
        return species if species is not None else []

    def density_species(self, prefix='rho_'):
        return [field[len(prefix) :] for field in self.simulation.avail_fields if field.startswith(prefix)]

    def get_laser_frequency(self, iteration, polarization='x', method='max'):
        """
        Calculates laser frequency for a specific iteration.
        Uses openpmd_reader LPA diagnostics.
        """
        return self.simulation.get_main_frequency(iteration=iteration, pol=polarization, method=method) / self.ureg.s

    def get_laser_wavelength(self, iteration, polarization='x', method='max'):
        """
        Calculates laser wavelength for a specific iteration.
        Uses openpmd_reader LPA diagnostics.
        """
        return (2 * _np.pi * self.c / self.get_laser_frequency(iteration, polarization, method)).to('um')
