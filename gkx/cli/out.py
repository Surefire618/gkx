import click
from pathlib import Path
import xarray as xr

# if this is smaller than the number of files,
# we get extremely verbose (but ultimately harmless)
# warnings, see https://github.com/pydata/xarray/issues/7549
xr.set_options(file_cache_maxsize=512)


@click.group()
def out():
    """out"""


@out.command()
@click.argument("file", default=Path("trajectory/"), type=Path)
@click.option("-fc", "--fc_file", default=None, type=Path)
@click.option("-dmx", "--dmx_file", default=None, type=Path)
@click.option("-o", "--outfile", default="greenkubo.nc", type=Path)
@click.option("--outfolder", default=Path("."), type=Path)
@click.option(
    "--maxsteps", default=None, type=int, help="cut off dataset after maxsteps"
)
@click.option("--offset", default=None, type=int, help="start from offset")
@click.option("--interpolate", is_flag=True, help="interpolate to dense grid")
@click.option("--maxnq", default=20, type=int, help="interpolate max q density")
@click.option("--spacing", default=None, type=int, help="use only every nth step")
@click.option(
    "--freq", default=None, type=float,
    help="filter-window frequency in THz; bypasses VDOS auto-detection",
)
def gk(file, fc_file, dmx_file, outfile, outfolder, maxsteps, offset, interpolate, maxnq, spacing, freq):
    """perform greenkubo analysis for heat flux dataset in FILE"""
    from gkx import comms
    from gkmx import open_dataset
    from gkmx import get_kappa_interpolate

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

    if interpolate:
        outfile = outfile.parent / f"{outfile.stem}.interpolate.nc"

    if maxnq != 20:
        outfile = outfile.parent / f"{outfile.stem}.nq_{maxnq}.nc"

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

    ds_gk = get_kappa_interpolate(
        dataset,
        fc_file,
        dmx_file,
        interpolate,
        nq_max=maxnq,
        backend="jax",
        freq=freq,
    )

    reporter.step(f"write to {outfile}")

    ds_gk.to_netcdf(outfile)

    reporter.done()

