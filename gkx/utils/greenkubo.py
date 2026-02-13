import json
import os
from pathlib import Path
from typing import Optional
import numpy as np
import xarray as xr
import collections
import scipy.signal as sl
from scipy.fft import fft, ifft, next_fast_len
import scipy.optimize as so

from ase import Atoms
from ase.geometry import find_mic
from ase import units

from vibes.io import parse_force_constants
from vibes.dynamical_matrix import DynamicalMatrix
from vibes import dimensions as dims
from vibes import keys, defaults
from vibes.brillouin import get_symmetrized_array
from vibes.helpers import talk, warn, Timer
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

n_threads = int(os.environ.get("OPENBLAS_NUM_THREADS", "1"))

def _talk(msg, **kw):
    """wrapper for `utils.talk` with prefix"""
    return talk(msg, prefix=_prefix, **kw)

def get_kappa_interpolate(
    dataset,
    fc_file: Optional[Path] = None,
    dmx_file: Optional[Path] = None,
    nq_max: int = 20,
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

    dmx_path = Path(dmx_file) if dmx_file is not None else None
    fc_path  = Path(fc_file)  if fc_file  is not None else None
    dmx = None

    if dmx_path is not None and dmx_path.exists():
        _talk(f"Load DynamicalMatrix from {dmx_path}")
        dmx = DynamicalMatrix.from_hdf5(str(dmx_path))
        dataset.update({keys.fc: (dims.fc, np.asarray(dmx.fc_phonopy))})

    elif fc_path is not None and fc_path.exists():
        msg = "Set up DynamicalMatrix"
        timer = Timer(msg)

        fc = parse_force_constants(str(fc_path), two_dim=False)

        dmx = DynamicalMatrix(
            force_constants=np.asarray(fc),
            primitive=primitive,
            supercell=supercell,
            with_group_velocity_matrices=True,   # QHGK needs it
        )

        if dmx_path is not None:
            _talk(f"Save DynamicalMatrix to {dmx_path}")
            dmx.to_hdf5(
                str(dmx_path),
                include_D_qij=True,
                include_group_velocity_matrices=True,
            )

        timer(msg)

    else:
        dmx = None

    if dmx is not None:
        dataset.update({keys.fc: (dims.fc, np.asarray(dmx.fc_phonopy))})

        dataset.update({keys.fc_remapped: (dims.fc_remapped, np.asarray(dmx.remapped))})

        map_s2p = np.asarray(dmx.I2iL_map[:, 0])
        dataset.attrs.update({keys.map_supercell_to_primitive: map_s2p})

    ds_gk = get_gk_dataset(
        dataset,
        dmx=dmx,
        interpolate=True,
        quasi_harmonic_greenkubo=True,
        nq_max=nq_max,
    )

    return ds_gk

def get_gk_dataset(
    dataset: xr.Dataset,
    dmx: DynamicalMatrix = None,
    interpolate: bool = False,
    quasi_harmonic_greenkubo: bool = False,
    nq_max: int = 20,
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
    if dmx is not None:

        data_ha = get_gk_interpolate(
            dataset,
            dmx=dmx,
            interpolate=interpolate,
            nq_max=nq_max,
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
    g_tsq = get_autocorrelation_diagonal(a_tsq)
    g2_tsq = (g_tsq * g_tsq.conj())

    # get lifetimes from fitting exp. function to mode amplitude
    tau_sq = get_lifetimes(g2_tsq.rename({keys.time_GKMA: keys.time}))

    # filter zero frequency lifetime
    tau_sq *= np.where(dmx.w_sq < 1.0e-6, np.nan, 1)

    return cv_sq, tau_sq

def get_a_tsq(
    U_tIa: np.ndarray,
    V_tIa: np.ndarray,
    masses: np.ndarray,
    e_sqI: np.ndarray,
    w_inv_sq: np.ndarray,
    stride: int = 1,
) -> np.ndarray:
    """get mode amplitude in shape [Nt, Ns, Nq]

    Formulation:
        a_tsq = 1/2 * (u_tsq + 1/iw_sq p_tsq)

    Args:
        U_tIa:    displacements as [time, atom_index, cartesian_index]
        V_tIa:    velocities    as [time, atom_index, cartesian_index]
        masses:   atomic masses as [atom_index]
        e_sqI: mode eigenvector as [band_index, q_point_index, atom_and_cartesian_index]
        w_inv_sq: inverse frequencies as [band_index, q_point_index]
        stride: use every STRIDE timesteps

    Returns:
        a_tsq: time resolved mode amplitudes

    """
    Nt = len(U_tIa[::stride])  # no. of timesteps
    # mass-weighted coordinates
    u_tI = np.asarray(masses[None, :, None] ** 0.5 * U_tIa[::stride]).reshape([Nt, -1])
    p_tI = np.asarray(masses[None, :, None] ** 0.5 * V_tIa[::stride]).reshape([Nt, -1])

    # project on modes
    u_tsq = np.moveaxis(e_sqI @ u_tI.T, -1, 0)
    p_tsq = np.moveaxis(e_sqI @ p_tI.T, -1, 0)

    # complex amplitudes
    a_tsq = 0.5 * (u_tsq + 1.0j * w_inv_sq[None, :] * p_tsq)

    return a_tsq

def get_autocorrelation_diagonal(
    array: xr.DataArray,
    verbose: bool = True,
    **kwargs
) -> xr.DataArray:
    """compute autocorrelation function for each componetn in  multi-dimensional xarray

    Args:
        array [N_t, *dims]: data with time axis in the front
    Returns:
        xarray.DataArray [N_t, dims]: autocorrelation along axis=0

    """
    msg = f"Get autocorrelation for array of shape {array.shape}"
    timer = Timer(msg, verbose=verbose)

    # memorize dimensions and shape of input array
    Nt, *shape = np.shape(array)

    # compute autocorrelation for each dimension
    # move time axis to last index
    data = np.moveaxis(np.asarray(array), 0, -1)
    corr = np.zeros((*shape, Nt), dtype=data.dtype)
    for Ia in np.ndindex(*shape):
        tmp = correlate(data[Ia], data[Ia], **kwargs)
        corr[Ia] = tmp
    # move time axis back to front
    corr = np.moveaxis(corr, -1, 0)

    da_corr = array.copy()
    da_corr.data = corr
    da_corr.name = f"{array.name}_{keys.autocorrelation}"

    timer()

    return da_corr

def correlate(f1, f2, normalize=2, hann=True):
    """Compute correlation function for signal f1 and signal f2

    Reference:
        https://gitlab.com/flokno/python_recipes/-/blob/master/mathematics/
        correlation_function/autocorrelation.ipynb

    Args:
        f1: signal 1
        f2: signal 2
        normalize: no (0), by length (1), by lag (2)
        hann: apply Hann window
    Returns:
        the correlation function
    """
    a1, a2 = (np.asarray(f) for f in (f1, f2))
    Nt = min(len(a1), len(a2))

    if Nt != max(len(a1), len(a2)):
        msg = "The two signals are not equally long: "
        msg += f"len(a1), len(a2) = {len(a1)}, {len(a2)}"
        warn(msg)

    corr = sl.correlate(a1[:Nt], a2[:Nt])[Nt - 1 :]

    if normalize is True or normalize == 1:
        corr /= Nt
    elif normalize == 2:
        corr /= np.arange(Nt, 0, -1)

    if hann:
        corr *= _hann(Nt)

    return corr

def _hann(nsamples: int):
    """Return one-side Hann function

    Args:
        nsamples (int): number of samples
    """
    return sl.windows.hann(2 * nsamples)[nsamples:]


def get_gk_interpolate(
    dataset: xr.Dataset,
    dmx: DynamicalMatrix = None,
    interpolate: bool = False,
    quasi_harmonic_greenkubo: bool = False,
    nq_max: int = 20,
):
    timer = Timer("Set up DynamicalMatrix")

    need_vssq = bool(quasi_harmonic_greenkubo)
    have_dmx = dmx is not None

    if have_dmx:
        sol = getattr(dmx, "solution", None)
        have_sol = sol is not None
        have_vssq = have_sol and (getattr(sol, "v_ssqa_cartesian", None) is not None)

        if need_vssq and (not have_vssq):
            dmx = DynamicalMatrix.from_dataset(dataset, with_group_velocity_matrices=True)
    else:
        dmx = DynamicalMatrix.from_dataset(
            dataset,
            with_group_velocity_matrices=need_vssq,
        )

    timer("Set up DynamicalMatrix")

    # heat capacity and phonon lifetime
    # cv_sq, tau_sq = get_flux_mode_data(dataset=dataset, dmx=dmx)
    timer = Timer(f"Compute_cv_tau using {n_threads} threads")
    cv_sq, tau_sq = compute_cv_tau(dataset=dataset, dmx=dmx, fft_workers=n_threads)
    timer("Compute_cv_tau with time")

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
            nq_max=nq_max,
            quasi_harmonic_greenkubo=quasi_harmonic_greenkubo
        )
        data.update(results)

    return collections.namedtuple("gk_ha_q_data", data.keys())(**data)

def compute_cv_tau(
    dataset,
    dmx,
    stride: int = 1,
    t_chunk: int = 2000,
    mode_block: int = 64,
    dtype_u: np.dtype = np.float32,
    dtype_a: np.dtype = np.complex64,
    tau_thresh: float = 0.1,
    hann: bool = True,
    normalize: int = 2,
    correct_finite_time: bool = True,
    fft_workers = None,
):
    """
    In-memory computation of:
      - cv_sq from Var(E) without storing E_tsq
      - tau_sq from g2(t)=|<a*(0)a(t)>|^2 via FFT ACF + log-linear fit
    No disk IO. No (Nt×Nmodes) arrays.

    Assumes:
      e_sqI shape = (s, q, I), I = 3*Natoms
      w_inv_sq, w2_sq shape = (s, q)
    """

    # ---------------- sizes & flatten ----------------
    # time length after stride
    Nt = dataset.displacements.isel({dataset.displacements.dims[0]: slice(None, None, stride)}).shape[0]
    dt = float(dataset.displacements.time[1] - dataset.displacements.time[0])

    e_sqI = np.asarray(dmx.e_sqI)  # (s,q,I)
    s, q, I = e_sqI.shape
    nmodes = s * q

    e_mI = e_sqI.reshape(nmodes, I).astype(dtype_a, copy=False)       # (mode, I)
    w_inv_m = np.asarray(dmx.w_inv_sq).reshape(nmodes).astype(np.float32, copy=False)
    w2_m = np.asarray(dmx.w2_sq).reshape(nmodes).astype(np.float32, copy=False)

    # ---------------- build mass-weighted u_I(t), p_I(t) once ----------------
    m = np.asarray(dataset.masses).astype(dtype_u, copy=False)              # (Natoms,)
    m_sqrt = np.sqrt(m).astype(dtype_u, copy=False)                         # (Natoms,)

    u_I_t = np.empty((I, Nt), dtype=dtype_u)  # (I, Nt)
    p_I_t = np.empty((I, Nt), dtype=dtype_u)

    # infer time dim name from dataset
    time_dim = dataset.displacements.dims[0]

    col = 0
    for raw0 in range(0, Nt, t_chunk):
        raw1 = min(Nt, raw0 + t_chunk)
        # map to original indices with stride
        src0 = raw0 * stride
        src1 = raw1 * stride

        U = np.asarray(dataset.displacements.isel({time_dim: slice(src0, src1, stride)}).data, dtype=dtype_u)
        V = np.asarray(dataset.velocities.isel({time_dim: slice(src0, src1, stride)}).data, dtype=dtype_u)
        tc = U.shape[0]

        # mass weight and flatten: (tc, Natoms, 3) -> (tc, I)
        # avoid creating gigantic temporaries; tc is small
        Uw = (U * m_sqrt[None, :, None]).reshape(tc, I)
        Vw = (V * m_sqrt[None, :, None]).reshape(tc, I)

        u_I_t[:, raw0:raw1] = Uw.T
        p_I_t[:, raw0:raw1] = Vw.T

        col += tc

    # ---------------- precompute normalization/window for ACF ----------------
    # autocorr length we compute (full Nt lags)
    nfft = next_fast_len(2 * Nt)
    if normalize == 2:
        norm = np.arange(Nt, 0, -1, dtype=np.float64)  # (Nt,)
    elif normalize is True or normalize == 1:
        norm = float(Nt)
    else:
        norm = None

    if hann:
        hann_vec = sl.windows.hann(2 * Nt)[Nt:].astype(np.float64)  # (Nt,)

    Tmax = (Nt - 1) * dt  # for finite-time correction

    # outputs in flattened mode axis
    cv_mode = np.empty(nmodes, dtype=np.float64)
    tau_mode = np.full(nmodes, np.nan, dtype=np.float64)

    # ---------------- process by mode blocks ----------------
    x_full = np.arange(Nt, dtype=np.float64)

    for m0 in range(0, nmodes, mode_block):
        m1 = min(nmodes, m0 + mode_block)
        B = m1 - m0

        e_blk = np.ascontiguousarray(e_mI[m0:m1, :])                 # (B, I), complex64
        w_inv_blk = w_inv_m[m0:m1].astype(np.float32, copy=False)    # (B,)
        w2_blk = w2_m[m0:m1].astype(np.float32, copy=False)          # (B,)

        # projections: (B,I) @ (I,Nt) -> (B,Nt)
        u_proj = e_blk @ u_I_t    # complex
        p_proj = e_blk @ p_I_t    # complex

        # a_blk in-place into u_proj to save memory:
        # a = 0.5 * (u + 1j*w_inv*p)
        p_proj *= (1.0j * w_inv_blk[:, None])
        u_proj += p_proj
        u_proj *= 0.5
        a_blk = u_proj  # (B,Nt) complex
        del p_proj, u_proj

        # -------- std(a) using Var = E(|a|^2) - |E(a)|^2 (no big temporaries) --------
        mean_a = a_blk.mean(axis=1)  # (B,)
        abs2 = (a_blk.real.astype(np.float32)**2 + a_blk.imag.astype(np.float32)**2)  # (B,Nt) float32
        mean_abs2 = abs2.mean(axis=1).astype(np.float64)  # (B,)
        mean_a_abs2 = (mean_a.real.astype(np.float64)**2 + mean_a.imag.astype(np.float64)**2)
        var_a = np.maximum(mean_abs2 - mean_a_abs2, 0.0)
        std_a = np.sqrt(var_a)  # (B,)

        # guard against tiny std_a
        std_min = 1e-8
        ok = std_a > std_min
        if not np.all(ok):
            # keep those modes as nan in tau; cv still computable but likely 0
            std_a = np.where(ok, std_a, 1.0)

        # -------- cv from Var(E) without building E --------
        # E = 2*w2*|a|^2
        # Var(E) = (2*w2)^2 * (E[|a|^4] - (E[|a|^2])^2)
        m1_abs2 = mean_abs2  # E[|a|^2]
        m2_abs2 = (abs2 * abs2).mean(axis=1).astype(np.float64)  # E[|a|^4]
        scale = (2.0 * w2_blk.astype(np.float64))               # (B,)
        var_E = (scale * scale) * (m2_abs2 - m1_abs2 * m1_abs2)
        # will divide by kB T^2 V outside after all blocks (need volume, temperature)

        cv_mode[m0:m1] = var_E

        # -------- normalize a for autocorr (match your a_tsq /= std) --------
        a_blk = (a_blk / std_a[:, None]).astype(dtype_a, copy=False)

        # -------- ACF via FFT: C = ifft(fft(a)*conj(fft(a)))[:Nt] --------
        F = fft(a_blk, n=nfft, axis=1, workers=fft_workers)
        S = F * np.conj(F)
        C = ifft(S, axis=1, workers=fft_workers)[:, :Nt]  # (B,Nt)
        del F, S

        # normalization + hann (match your correlate())
        if norm is not None:
            if normalize == 2:
                C = C / norm[None, :]
            else:
                C = C / norm
        if hann:
            C = C * hann_vec[None, :]

        g2 = (C.real.astype(np.float64)**2 + C.imag.astype(np.float64)**2)  # (B,Nt), real
        del C

        # -------- lifetime fit (log-linear), mimic your threshold logic --------
        for i in range(B):
            tau_mode[m0 + i] = fit_tau_curvefit_from_g2(g2[i], dt=dt, thresh=0.5)
            # tau_mode[m0 + i] = fit_tau_loglin_weighted(g2[i], dt)

    # ---------------- finalize cv with kB, T, V ----------------
    volume = float(np.asarray(dataset.volume).mean())
    temperature = float(np.asarray(dataset.temperature).mean())
    cv_mode = cv_mode / units.kB / (temperature**2) / volume

    if correct_finite_time:
        tau_mode = np.asarray(tau_mode, dtype=np.float64)
        good = np.isfinite(tau_mode) & (tau_mode > 0)
        tau_mode_corr = np.full_like(tau_mode, np.nan)
        tau_mode_corr[good] = 1.0 / (1.0 / tau_mode[good] - 1.0 / Tmax)
        tau_mode = tau_mode_corr

    # reshape back to (s,q)
    cv_sq = cv_mode.reshape(s, q)
    tau_sq = tau_mode.reshape(s, q)

    # filter zero frequency lifetime
    tau_sq *= np.where(dmx.w_sq < 1.0e-6, np.nan, 1)

    # convert to xd.DataArray
    sq = (dims.s, dims.q)
    cv_sq = xr.DataArray(cv_sq, dims=sq, name=keys.mode_heat_capacity)
    tau_sq = xr.DataArray(tau_sq, dims=sq, name=keys.mode_lifetime)
    return cv_sq, tau_sq

def _exp(x, tau, y0):
    return y0 * np.exp(-x / tau)

def fit_tau_curvefit_from_g2(g2_1d, dt, thresh=0.1, maxfev=2000):
    """
    g2_1d: shape (Nt,), real
    Returns tau in time unit (tau_index * dt)
    Implements your original logic:
      - find first index where corr < 0.1
      - fit [0:first_drop) with y0*exp(-x/tau)
    """
    y = np.asarray(g2_1d, dtype=np.float64)
    if not np.isfinite(y[0]) or y[0] < 1e-12:
        return np.nan

    idx = np.where(y < thresh)[0]
    if idx.size == 0 or idx.min() < 2:
        return np.nan

    first_drop = int(idx.min())
    x = np.arange(first_drop, dtype=np.float64)
    yy = y[:first_drop]

    yy_clip = np.clip(yy, 1e-300, None)
    slope = np.polyfit(x, np.log(yy_clip), 1)[0]
    tau0 = (-1.0 / slope) if slope < 0 else max(2.0, first_drop / 5.0)
    y00 = float(yy[0])

    try:
        (tau_hat, y0_hat), _ = so.curve_fit(
            _exp,
            x, yy,
            p0=(tau0, y00),
            bounds=([1e-12, 1e-300], [np.inf, np.inf]),
            maxfev=maxfev,
        )
        return float(tau_hat) * float(dt)
    except Exception:
        return np.nan

def fit_tau_loglin_weighted(y, dt, drop_thresh=0.1, fit_floor=0.3):
    """
    y: g2(t) 1d array
    dt: timestep (already includes stride)
    drop_thresh: define first_drop where y < drop_thresh
    fit_floor: only fit points with y >= fit_floor to reduce tail influence
    """
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y[0]) or y[0] < 1e-12:
        return np.nan

    idx = np.where(y < drop_thresh)[0]
    if idx.size == 0 or idx.min() < 2:
        return np.nan
    n = int(idx.min())

    x = np.arange(n, dtype=np.float64)
    yy = y[:n]
    m = yy >= fit_floor
    if m.sum() < 3:
        # fallback: if too few points above fit_floor, use all up to first_drop
        m = np.ones_like(yy, dtype=bool)

    xx = x[m]
    yyy = np.clip(yy[m], 1e-300, None)
    ly = np.log(yyy)

    # weights: emphasize large y, suppress tail
    w = yyy * yyy
    W = w.sum()
    xbar = (w * xx).sum() / W
    ybar = (w * ly).sum() / W
    denom = (w * (xx - xbar) ** 2).sum()
    if denom == 0:
        return np.nan

    slope = (w * (xx - xbar) * (ly - ybar)).sum() / denom
    if slope >= 0:
        return np.nan

    return (-1.0 / slope) * float(dt)


