import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display

import copy
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroplan.plots import plot_sky
from astroplan import FixedTarget
from . import utils
from .const import COLORS
from . import pointing

from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import SphericalCircle

from shapely.geometry import mapping
from descartes import PolygonPatch

def display_1d(table, proj, ax=None, labels=None, **kwargs):
    xb = utils.calc_mean(table, proj[0])
    
    ax = plt.figure().add_subplot(111) if ax is None else ax

    for i, [tels, label] in enumerate(zip(table.groups, labels)):
        c = COLORS(i)
        for val in tels[proj[0]]:
            ax.axvline(val, label=label, color=c, **kwargs)
            label='_nolegend_'

    ax.axvline(xb, color="r", label='barycenter', **kwargs)
    ax.set_xlabel(f"{proj[0]} [m]")
    ax.set_yticks([0, 1])
    ax.legend(frameon=False)

    return ax

def display_2d(table, proj, ax=None, labels=None, **kwargs):
    if ax is None:
        ax = plt.figure().add_subplot(111)
            
    scale = 1
    
    b_output = utils.calc_mean(table, [proj[0], proj[1], f"p_{proj[0]}", f"p_{proj[1]}"])
    
    for i, [tels, label] in enumerate(zip(table.groups, labels)):
        xx = tels[proj[0]]
        yy = tels[proj[1]]
        xv = tels[f"p_{proj[0]}"]
        yv = tels[f"p_{proj[1]}"]
        ids = tels["id"]

        s = ax.scatter(xx, yy, label=label, **kwargs)
        ax.quiver(xx, yy, xv, yv, color=s.get_facecolor())

        for i, x, y in zip(ids, xx, yy):
            ax.annotate(i, (x, y))
    
    ax.scatter(b_output[0], b_output[1], marker='+', label='barycenter', color="r")
    ax.quiver(*b_output, color="r")
    ax.set_xlabel(f"{proj[0]} [m]")
    ax.set_ylabel(f"{proj[1]} [m]")

    ax.grid('on')
    ax.axis('equal')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] - 0.25 * np.abs(xlim[0]), xlim[1] + 0.25 * np.abs(xlim[1]))
    ax.set_ylim(ylim[0] - 0.25 * np.abs(ylim[0]), ylim[1] + 0.25 * np.abs(ylim[1]))
    ax.legend(frameon=False)

    return ax

def display_3d(table, proj, ax=None, labels=None, **kwargs):
    ax = plt.figure().add_subplot(111, projection='3d')

    scale = 1

    max_range = []
    for axis in ["x", "y", "z"]:
        max_range.append(table[axis].max() - table[axis].min())
    max_range = max(max_range)

    for i, [tels, label] in enumerate(zip(table.groups, labels)):
        xx = tels["x"]
        yy = tels["y"]
        zz = tels["z"]
        c = COLORS(i)
        ax.quiver(xx, yy, zz, 
                tels["p_x"], tels["p_y"], tels["p_z"],
                length=max_range,
                label=label,
                color=c,
                )

        Xb = scale * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + scale * (xx.max() + xx.min())
        Yb = scale * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + scale * (yy.max() + yy.min())
        Zb = scale * max_range * np.mgrid[-0.01:2:2, -0.01:2:2, -0.01:2:2][2].flatten() + scale * (zz.max() + zz.min())
        
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        
    xx = utils.calc_mean(table, proj[0])
    yy = utils.calc_mean(table, proj[1])
    zz = utils.calc_mean(table, proj[2])
    xbv = utils.calc_mean(table, f"p_{proj[0]}")
    ybv = utils.calc_mean(table, f"p_{proj[1]}")
    zbv = utils.calc_mean(table, f"p_{proj[2]}")

    ax.quiver(xx, yy, zz, 
            xbv, ybv, zbv,
            color="r",
            length=max_range,
            label='barycenter',
            )
     
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.legend(frameon=False)
    
    return ax

def display_barycenter(table, proj, ax=None, labels=None, fig=None, **kwargs):
    if fig is None:
        fig = plt.figure() 

    if ax is None:
        ax = fig.add_subplot(111) 

    scale = 1
    
    for i, (tab, label) in enumerate(zip(table.groups, labels)):
        output = utils.calc_mean(tab, [proj[0], proj[1], f"p_{proj[0]}", f"p_{proj[1]}"])
        s = ax.scatter(output[0], output[1], color=COLORS(i),)
        ax.quiver(*output, color=s.get_facecolor())

        ax.annotate(label, (output[0], output[1]))

    ax.set_xlabel(f"{proj[0]} [m]")
    ax.set_ylabel(f"{proj[1]} [m]")

    ax.grid('on')
    ax.axis('equal')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0] - 0.25 * np.abs(xlim[0]), xlim[1] + 0.25 * np.abs(xlim[1]))
    ax.set_ylim(ylim[0] - 0.25 * np.abs(ylim[0]), ylim[1] + 0.25 * np.abs(ylim[1]))
    ax.legend(frameon=False)

    return ax

def interactive_barycenter(array, proj="xy", overwrite=True, group=False):
    if overwrite:
        new_array = array
    else:
        new_array = copy.deepcopy(array)
    
    fig = plt.figure()

    def update(div=0, az=0, alt=0):
        new_array.divergent_pointing(div, az=az, alt=alt, units='deg')
        new_array.__convert_units__(toDeg=True)
        plt.cla()
        grouped_table, labels = new_array.group_by(group)
        display_barycenter(grouped_table, proj, labels=labels, fig=fig)
        fig.canvas.draw_idle()

    div_s = widgets.FloatLogSlider(value=new_array.div, base=10, min=-4, max=0, step=0.2, description='Divergence')
    az_s = widgets.FloatSlider(value=new_array.pointing["az"].value, min=0, max=360, step=0.01, description='Azimuth [deg]')
    alt_s = widgets.FloatSlider(value=new_array.pointing["alt"].value, min=0, max=90, step=0.01, description='Altitude [deg]')
    
    ui = widgets.HBox([div_s, alt_s, az_s])
    out = widgets.interactive_output(update, {'div': div_s, 'az': az_s, 'alt': alt_s})
    display(ui, out)

    return new_array

def display_skymap(table, frame, ax=None, **kwargs):
    ax = plt.figure().add_subplot(111, projection='polar') if ax is None else ax

    radec = pointing.pointing_coord(table, frame, icrs=True)
    point = SkyCoord(ra=radec.ra, dec=radec.dec)
    target = FixedTarget(coord=point, name="source")

    plot_sky(target, frame.observer, frame.t_obs, ax=ax, style_kwargs=kwargs)

    return ax

def skymap_polar(array, group=False, fig=None, filename=None):
    if fig is None:
        fig = plt.figure() 
    else:
        ax = fig.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    tr = Affine2D().scale(np.pi/180., 1.).translate(+np.pi/2., 0) + plt.polar().get_transform()

    n = 20
    extreme_finder = angle_helper.ExtremeFinderCycle(10, 10,
                                                     lon_cycle=360,
                                                     lat_cycle=None,
                                                     lon_minmax=None,
                                                     lat_minmax=(-90, 90),
                                                     )

    grid_locator1 = angle_helper.LocatorDMS(12)

    tick_formatter1 = angle_helper.FormatterDMS()

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )

    ax1 = plt.subplot(projection=grid_helper)

    array.__convert_units__(toDeg=True)
    tel_group, labels = array.group_by(group)

    for tel_table, label in zip(tel_group.groups, labels):
        s = ax1.scatter(tel_table["az"], tel_table["alt"], label=label,
            s=20, edgecolor="black", transform=ax1.transData, zorder=10)
        
        for tel in tel_table:
            r = SphericalCircle((tel["az"] * u.deg, tel["alt"] * u.deg), tel["radius"] * u.deg, 
                                color=s.get_facecolor()[0], alpha=0.1, transform=ax1.transData)
            ax1.add_patch(r)
            ax1.annotate(tel["id"], (tel["az"], tel["alt"]), fontsize=12, xytext=(4, 4), 
                color="black", textcoords='offset pixels', zorder=10)

    ax1.grid(True)
    ax1.set_xlabel("Azimuth [deg]", fontsize=20)
    ax1.set_ylabel("Altitude [deg]", fontsize=20)
    ax1.legend(loc=1)

    if filename is not None:
        plt.savefig(filename)
        plt.show(block=False)

def interactive_polar(array, overwrite=True, group=False):
    if overwrite:
        new_array = array
    else:
        new_array = copy.deepcopy(array)

    fig = plt.figure()

    def update(div=0, az=0, alt=0):
        new_array.divergent_pointing(div, az=az, alt=alt, units='deg')
        new_array.__convert_units__(toDeg=True)
        plt.cla()
        new_array.skymap_polar(group=group, fig=fig)
        fig.canvas.draw_idle()

    div_s = widgets.FloatLogSlider(value=new_array.div, base=10, min=-4, max=0, step=0.2, description='Divergence')
    az_s = widgets.FloatSlider(value=new_array.pointing["az"].value, min=0, max=360, step=0.01, description='Azimuth [deg]')
    alt_s = widgets.FloatSlider(value=new_array.pointing["alt"].value, min=0, max=90, step=0.01, description='Altitude [deg]')
    
    ui = widgets.HBox([div_s, alt_s, az_s])
    out = widgets.interactive_output(update, {'div': div_s, 'az': az_s, 'alt': alt_s})
    display(ui, out)

    return new_array


def multiplicity_plot(array, m_cut=0, fig=None):

    m, overlaps, geoms = array.hFoV(m_cut=m_cut, full_output=True)
    max_m = int(array.size_of_array)
    ave_multi = np.average(m[:, 0], weights=m[:, 1])
    var_multi = np.average((m[:, 0] - ave_multi) ** 2, weights=m[:, 1])

    if fig is None:
        fig, (ax, ax_mul) = plt.subplots(1, 2, figsize=(10, 4))
        ax_cb = fig.add_axes([0.44, 0.15, 0.01, 0.7])

    cmap = plt.cm.get_cmap('rainbow')
    color_list = cmap(np.linspace(0, 1, max_m))
    bounds = np.arange(max_m + 1) + 1

    ax_cb = fig.add_axes([0.44, 0.15, 0.01, 0.7])

    minmax = []
    for i, pol in enumerate(geoms):
        colore = int(overlaps[i])
        pol_map = mapping(pol)
        ax.add_patch(PolygonPatch(pol_map, color=color_list[colore - 1]))
        patch_az = np.asarray(pol_map['coordinates'])[0][:][0]
        minmax.append([min(patch_az), max(patch_az)])
    minmax = np.asarray(minmax)

    norm = plt.colors.BoundaryNorm(bounds, cmap.N)

    cb1 = mpl.colorbar.ColorbarBase(ax_cb,
                                    norm=norm,
                                    cmap=cmap,
                                    boundaries=bounds,
                                    orientation='vertical',
                                    label='Multiplicity')
    cb1.set_ticks(np.arange(0, max_m + 1, step=4) + 1)
    cb1.set_ticklabels(np.arange(0, max_m + 1, step=4))

    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Altitude [deg]")
    ax.set_xlim(np.min(array.table["az"]) - 5, np.max(array.table["az"]) + 5)
    ax.set_ylim(np.min(array.table["alt"]) - 5, np.max(array.table["alt"]) + 5)

    ax.text(0.9, 0.9, r"Average: {:.1f} $\pm$ {:.1f}".format(ave_multi, np.sqrt(var_multi)),
            ha="right", transform=ax.transAxes)

    ax_mul.bar(m[:, 0], m[:, 1])
    ax_mul.text(0.9, 0.9, "Total hFoV = {:.0f}".format(sum(m[:, 1][m[:, 0] >= m_cut])),
                ha="right", transform=ax_mul.transAxes)
    ax_mul.set_xticks(np.arange(0, max_m + 1, step=2))
    ax_mul.set_xlim(0.5, max_m + 0.5)
    ax_mul.set_ylabel('HFOV')
    ax_mul.set_xlabel('Multiplicity')

    plt.show()

def interactive_multiplicity(array, overwrite=True):

    if overwrite:
        new_array = array
    else:
        new_array = copy.deepcopy(array)

    fig, (ax, ax_mul) = plt.subplots(1, 2, figsize=(10, 4))
    ax_cb = fig.add_axes([0.44, 0.15, 0.01, 0.7])

    def update(div=0, az=0, alt=0):

        new_array.divergent_pointing(div, az=az, alt=alt, units='deg')
        new_array.__convert_units__(toDeg=True)
        plt.cla()
        new_array.multiplicity_plot(fig=(fig, (ax, ax_mul)))
        fig.canvas.draw_idle()

    div_s = widgets.FloatLogSlider(value=new_array.div, base=10, min=-4, max=0, step=0.2, description='Divergence')
    az_s = widgets.FloatSlider(value=new_array.pointing["az"].value, min=0, max=360, step=0.01,
                               description='Azimuth [deg]')
    alt_s = widgets.FloatSlider(value=new_array.pointing["alt"].value, min=0, max=90, step=0.01,
                                description='Altitude [deg]')

    ui = widgets.HBox([div_s, alt_s, az_s])
    out = widgets.interactive_output(update, {'div': div_s, 'az': az_s, 'alt': alt_s})
    display(ui, out)

    return new_array
