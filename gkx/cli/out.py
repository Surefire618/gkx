import click
from pathlib import Path
import xarray as xr
import json
import numpy as np

# if this is smaller than the number of files,
# we get extremely verbose (but ultimately harmless)
# warnings, see https://github.com/pydata/xarray/issues/7549
xr.set_options(file_cache_maxsize=512)

from vibes import keys

@click.group()
def out():
    """out"""


@out.command()
@click.argument("file", default=Path("trajectory/"), type=Path)
@click.option("-fc", "--fc_file", default=None, type=Path)
@click.option("-o", "--outfile", default="greenkubo.nc", type=Path)
@click.option("--outfolder", default=Path("."), type=Path)
@click.option(
    "--maxsteps", default=None, type=int, help="cut off dataset after maxsteps"
)
@click.option("--offset", default=None, type=int, help="start from offset")
@click.option("--interpolate", is_flag=True, help="interpolate to dense grid")
@click.option("--spacing", default=None, type=int, help="use only every nth step")
@click.option("--freq", default=1.0, type=float, help="lowest characteristic frequency")
def gk(file, fc_file, outfile, outfolder, maxsteps, offset, interpolate, spacing, freq):
    """perform greenkubo analysis for heat flux dataset in FILE"""
    from stepson import comms
    from stepson.green_kubo import get_kappa_dataset
    from stepson.utils import open_dataset

    reporter = comms.reporter()
    reporter.start(f"working on {file}")

    outfolder.mkdir(exist_ok=True)
    outfile = outfolder / outfile

    if offset is not None:
        if offset < 0:
            dataset = open_dataset(file)
            offset = len(dataset.time) + offset

    if offset is not None:
        outfile = outfile.parent / f"{outfile.stem}.from_{offset}.nc"

    if maxsteps is not None:
        outfile = outfile.parent / f"{outfile.stem}.to_{maxsteps}.nc"

    if spacing is not None:
        outfile = outfile.parent / f"{outfile.stem}.every_{spacing}.nc"

    if freq is not None:
        outfile = outfile.parent / f"{outfile.stem}.freq_{freq:.2f}.nc"

    if outfile.is_file():
        comms.warn(f"{outfile} exists, skipping")
        reporter.done()
        return None

    dataset = open_dataset(file)

    if maxsteps is not None:
        comms.talk(f"truncating to {maxsteps} timesteps")

        if len(dataset.time) < maxsteps:
            comms.warn(
                f"Tried to truncate {len(dataset.time)} timesteps to {maxsteps}, but dataset too short."
            )
            reporter.done()
            return None

        dataset = dataset.isel(time=slice(0, maxsteps))

    if offset is not None:
        comms.talk(f"starting from timestep {offset}")
        dataset = dataset.isel(time=slice(offset, len(dataset.time)))

    if spacing is not None:
        comms.talk(f"using spacing {spacing}")
        dataset = dataset.isel(time=slice(0, len(dataset.time), spacing))

    if fc_file is not None and interpolate:
        ds_gk = get_kappa_interpolate(
            dataset,
            fc_file,
        )
    else:
        ds_gk = get_kappa_dataset(
            dataset,
            window_factor=1.0,
            aux=False,
            freq=freq,
        )

    reporter.step(f"write to {outfile}")

    ds_gk.to_netcdf(outfile)

    reporter.done()

def get_kappa_interpolate(
    dataset,
    fc_file,
):
    from vibes.green_kubo import get_gk_dataset
    from vibes.io import parse_force_constants
    from vibes.force_constants import ForceConstants
    from vibes import dimensions as dims
    from ase import Atoms
    from ase.geometry import find_mic

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
