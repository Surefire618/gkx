import json
import numpy as np
import xarray as xr
import collections

from ase import Atoms
from ase.geometry import find_mic
from ase import units

from vibes.io import parse_force_constants
from vibes.force_constants import ForceConstants
from vibes.dynamical_matrix import DynamicalMatrix
from vibes import dimensions as dims
from vibes import keys, defaults
from vibes.brillouin import get_symmetrized_array
from vibes.correlation import get_autocorrelationNd
from vibes.helpers import talk, warn
from vibes.green_kubo import (
    get_gk_dataset,
    get_gk_prefactor_from_dataset,
    get_hf_data,
    get_lowest_vibrational_frequency,
    get_filtered
)
from vibes.green_kubo.harmonic import (
    get_lifetimes,
    get_a_tsq,
    get_kappa,
)
from vibes.green_kubo.interpolation import get_interpolation_data

_prefix = "gk.interpolation"

def _talk(msg, **kw):
    """wrapper for `utils.talk` with prefix"""
    return talk(msg, prefix=_prefix, **kw)

def get_kappa_interpolate(
    dataset,
    fc_file,
):

    dataset[keys.heat_flux] /= 1000
    # dataset[keys.heat_flux_aux] /= 1000

    primitive = Atoms(**json.loads(dataset.attrs[keys.reference_primitive]))
    supercell = Atoms(**json.loads(dataset.attrs[keys.reference_supercell]))

    cell = np.asarray(supercell.cell)
    shape = dataset[keys.positions].shape

    displacements = dataset[keys.positions].data - supercell.positions
    displacements = find_mic(displacements.reshape(-1, 3), cell)[0]
    displacements = displacements.reshape(*shape)

    volumes = np.ones(shape[0]) * dataset.volume

    dataset.update({
        keys.displacements: (dims.time_atom_vec, displacements),
        keys.volume: (dims.time, volumes),
    })

    if fc_file:
        fc = parse_force_constants(fc_file, two_dim=False)
        fcs = ForceConstants(
            force_constants=fc, primitive=primitive, supercell=supercell
        )

        fc = fcs.array
        dataset.update({keys.fc: (dims.fc, fc)})
        rfc = fcs.remapped
        dataset.update({keys.fc_remapped: (dims.fc_remapped, rfc)})
        map_s2p = fcs.I2iL_map[:, 0]
        dataset.attrs.update({keys.map_supercell_to_primitive: map_s2p})

    ds_gk = get_gk_dataset(
        dataset,
        interpolate=True,
        quasi_harmonic_greenkubo=True,
    )

    return ds_gk


def get_gk_dataset(
    dataset: xr.Dataset,
    interpolate: bool = False,
    harmonic_crosscorrelation: bool = False,
    greenkubo_mode_analysis: bool = False,
    quasi_harmonic_greenkubo: bool = False,
    window_factor: int = defaults.window_factor,
    filter_prominence: float = defaults.filter_prominence,
    discard: int = 0,
    total: bool = False,
    cross_offdiag: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """get Green-Kubo data from trajectory dataset

    Args:
        dataset: a dataset containing `heat_flux` and describing attributes
        interpolate: interpolate harmonic flux to dense grid
        window_factor: factor for filter width estimated from VDOS (default: 1)
        filter_prominence: prominence for peak detection
        discard: discard this many timesteps from the beginning of the trajectory
        total: postprocess gauge-invariant terms of heat flux as well

    Returns:
        xr.Dataset: the processed data

    Workflow:
        1. get heat flux autocorrelation function (HFACF) and the integrated kappa
        2. get lowest significant vibrational frequency and get its time period
        3. filter integrated kappa with this period
        4. get HFACF by time derivative of this filtered kappa
        5. filter the HFACF with the same filter
        6. estimate cutoff time from the decay of the filtered HFACF
        7. run harmonic heat flux and interpolation
    """

    # 1. get HFACF and integrated kappa
    heat_flux = dataset[keys.heat_flux]

    if total:  # add non-gauge-invariant contribution
        if keys.heat_flux_aux in dataset:
            heat_flux += dataset[keys.heat_flux_aux]
        else:
            warn("Heat flux aux not in dataset!!!", level=1)

    # get prefactor V/kB/T**2
    gk_prefactor = get_gk_prefactor_from_dataset(dataset, verbose=verbose)

    # get heatflux and integrated kappa
    hfacf, kappa = get_hf_data(heat_flux, prefactor=gk_prefactor ,total=total)

    # 2. get lowest significant frequency (from VDOS) in THz
    kw = {"prominence": filter_prominence}
    freq = get_lowest_vibrational_frequency(dataset[keys.velocities], **kw)

    # window in fs from freq.:
    window_fs = window_factor / freq * 1000

    kw_talk = {"verbose": verbose}
    _talk("Estimate filter window size", **kw_talk)
    _talk(f".. lowest vibrational frequency: {freq:.4f} THz", **kw_talk)
    _talk(f".. corresponding window size:    {window_fs:.4f} fs", **kw_talk)
    _talk(f".. window multiplicator used:    {window_factor:.4f} fs", **kw_talk)

    # 3. filter integrated HFACF (kappa) with this window respecting antisymmetry in time
    kw = {"window_fs": window_fs, "antisymmetric": True, "verbose": verbose}
    k_filtered = get_filtered(kappa, **kw)

    # 4. get the respective HFACF by differentiating w.r.t. time and filtering again
    k_gradient = kappa.copy()

    # compute derivative with j = dk/dt = dk/dn * dn/dt = dk/dn / dt
    dt = float(kappa.time[1] - kappa.time[0])
    k_gradient.data = np.gradient(k_filtered, axis=0) / dt

    j_filtered = get_filtered(k_gradient, window_fs=window_fs, verbose=False)

    # 5. get cutoff times from j_filtered and save respective kappas
    ts = np.zeros([3, 3])
    ks = np.zeros([3, 3])

    # symmetrize J and kappa, together with the cutoff time
    j_filtered_sym = 0.5 * (j_filtered + np.swapaxes(j_filtered, 1, 2))
    k_filtered_sym = 0.5 * (k_filtered + np.swapaxes(k_filtered, 1, 2))

    # iterate diagonal first
    for ii, jj in np.array(list(np.ndindex(3, 3)))[[0, 4, 8, 1, 2, 3, 5, 6, 7], :]:
        j = j_filtered_sym[:, ii, jj]
        if ii == jj:
            times = j.time[j < 0]  # diagonal cutoff time
        else:
            if cross_offdiag:
                cross_time = (ts[ii, ii] + ts[jj, jj]) / 2
                times = j.time[j.time > cross_time]  # off-diagonal cutoff time
            else:
                times = j.time[j / j[0] < 0]  # off-diagonal cutoff time

        if len(times) > 1:
            ta = times.min()
        else:
            warn("no cutoff time found", level=1)
            ta = 0
        ks[ii, jj] = k_filtered_sym[:, ii, jj].sel(time=ta)
        ts[ii, jj] = ta

    k_diag = np.diag(ks)
    k_mean = np.mean(k_diag)
    k_err = np.std(k_diag) / 3**0.5

    # report
    if verbose:
        _talk(["Cutoff times (fs):", *np.array2string(ts, precision=3).split("\n")])
        _talk(f"Kappa is:       {k_mean:.3f} +/- {k_err:.3f} W/mK")
        _talk(["Kappa^ab (W/mK) is: ", *np.array2string(ks, precision=3).split("\n")])

    # 6. compile new dataset
    # add filter parameters to attrs
    attrs = dataset.attrs.copy()
    u = {
        keys.gk_window_fs: window_fs,
        keys.gk_prefactor: gk_prefactor,
        keys.filter_prominence: filter_prominence,
    }
    attrs.update(u)

    data = {
        keys.heat_flux: heat_flux,
        keys.hf_acf: hfacf,
        keys.hf_acf_filtered: j_filtered,
        keys.kappa_cumulative: kappa,
        keys.kappa_cumulative_filtered: k_filtered,
        keys.kappa: (dims.tensor, ks),
        keys.time_cutoff: (dims.tensor, ts),
    }

    # 7. add properties derived from harmonic model
    if keys.fc in dataset:

        data_ha = get_gk_interpolate(
            dataset,
            interpolate=interpolate,
            quasi_harmonic_greenkubo=quasi_harmonic_greenkubo,
        )

        data.update(data_ha._asdict())
        data.update({keys.fc: dataset[keys.fc]})

        if interpolate:
            correction = data_ha.interpolation_correction
            correction_ab = data_ha.interpolation_correction_ab[1]
            kappa_corrected = ks + correction * np.eye(3)
            kappa_corrected_ab = ks + correction_ab
            data.update({keys.kappa_corrected: (dims.tensor, kappa_corrected)})
            _talk("END RESULT: Finite-size corrected thermal conductivity")
            _talk(
                f"Corrected kappa is:       {k_mean+correction:.3f} +/- {k_err:.3f} W/mK"
            )
            _talk(
                [
                    "Corrected kappa^ab (W/mK) is: ",
                    *np.array2string(kappa_corrected_ab, precision=3).split("\n"),
                ]
            )

    # add thermodynamic properties
    data.update({key: dataset[key] for key in (keys.volume, keys.temperature)})

    return xr.Dataset(data, coords=kappa.coords, attrs=attrs)


def get_flux_mode_data(
    dataset: xr.Dataset, dmx: DynamicalMatrix, stride: int = 1
):
    """return harmonic flux data with flux and mode energies as xr.DataArrays

    Args:
        dataset: the trajectory dataset
        dmx: the dynamical matrix (object)
        stride: time step length

    Returns:
        (J_ha_q (flux), E_tsq (mode energy))

    """
    a_tsq = get_a_tsq(
        U_tIa=dataset.displacements,
        V_tIa=dataset.velocities,
        masses=dataset.masses,
        e_sqI=dmx.e_sqI,
        w_inv_sq=dmx.w_inv_sq,
        stride=stride,
    )

    # compute mode-resolved energy
    # E = 2 * w2 * a+.a
    E_tsq = 2 * abs(a_tsq) ** 2 * dmx.w2_sq
    volume = dataset.volume.mean().data

    # heat capacity
    volume = dataset.volume.mean().data
    temperature = dataset.temperature.mean().data
    cv_sq = E_tsq.var(axis=0) / units.kB / temperature**2 / volume
    sq = (dims.s, dims.q)
    cv_sq = xr.DataArray(cv_sq, dims=sq, name=keys.mode_heat_capacity)

    # make dataarrays incl. coords and dims
    tsq = (keys.time_GKMA, dims.s, dims.q)
    coords = {keys.time_GKMA: dataset.coords[keys.time][::stride].data}
    a_tsq = xr.DataArray(a_tsq, coords=coords, dims=tsq, name="a_tsq")

    # compute <a^+(t) a(0)>
    a_std_sq = a_tsq.data.std(axis=0)
    a_tsq /= a_std_sq[None,:,:]
    g_tsq = get_autocorrelationNd(a_tsq, off_diagonal=False)
    g2_tsq = (g_tsq * g_tsq.conj())

    # get lifetimes from fitting exp. function to mode amplitude
    tau_sq = get_lifetimes(g2_tsq.rename({keys.time_GKMA: keys.time}))

    return cv_sq, tau_sq

def get_gk_interpolate(
    dataset: xr.Dataset,
    dmx: DynamicalMatrix = None,
    interpolate: bool = False,
    quasi_harmonic_greenkubo: bool = False,
):

    if quasi_harmonic_greenkubo:
        dmx = DynamicalMatrix.from_dataset(dataset, with_group_velocity_matrices=True)
    else:
        if dmx is None:
            dmx = DynamicalMatrix.from_dataset(dataset)

    # heat capacity and phonon lifetime
    cv_sq, tau_sq = get_flux_mode_data(dataset=dataset, dmx=dmx)

    # symmetrize by averaging over symmetry-related q-points
    map2ir, map2full = dmx.q_grid.map2ir, dmx.q_grid.ir.map2full
    tau_symmetrized_sq = get_symmetrized_array(tau_sq, map2ir=map2ir, map2full=map2full)

    # gk prefactor
    gk_prefactor = get_gk_prefactor_from_dataset(dataset, verbose=False)

    # compute thermal concuctivity
    v_sqa = dmx.v_sqa_cartesian
    K_ha_q = get_kappa(v_sqa=v_sqa, tau_sq=tau_sq, cv_sq=cv_sq)
    K_ha_q.name = keys.kappa_ha

    # scalar cv: account for cv/kB not exactly 1 in the numeric simulation
    # choose such that c * K(c_s=kB) = K(c_s=c_s)
    k = K_ha_q.data.diagonal().mean()
    if k < 1.0e-4:  # comment: this criteria needs a careful tuning
        cv = cv_sq.mean()
    else:
        cv = k / get_kappa(v_sqa=v_sqa, tau_sq=tau_sq, scalar=True)
    cv = xr.DataArray(cv, name=keys.heat_capacity)

    # symmetrized thermal conductivity
    K_ha_q_symmetrized = get_kappa(v_sqa=v_sqa, tau_sq=tau_symmetrized_sq, cv_sq=cv)
    K_ha_q_symmetrized.name = keys.kappa_ha_symmetrized

    arrays = [K_ha_q, K_ha_q_symmetrized, cv_sq, cv, tau_sq, tau_symmetrized_sq]

    # add dynamical matrix arrays
    arrays += dmx._get_arrays()

    data = {ar.name: ar.astype("f4") for ar in arrays} # FIXME

    # interpolate
    if interpolate:
        results = get_interpolation_data(
            dmx=dmx,
            lifetimes=tau_symmetrized_sq,
            cv=cv,
            quasi_harmonic_greenkubo=quasi_harmonic_greenkubo
        )
        data.update(results)

    return collections.namedtuple("gk_ha_q_data", data.keys())(**data)
