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
#from . import pointing
from adjustText import adjust_text
from matplotlib.transforms import Affine2D
from astropy.visualization.wcsaxes import SphericalCircle
import itertools
import healpy as hp
import tqdm


def display_1d(table, proj, ax=None, labels=None, **kwargs):
    xb = utils.calc_mean(table, proj[0])
    print(table)
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
            print(xb, yb, zb)
        
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
def multiplicity_plot_2_div(array, array_2, subarray_mult_1=None, subarray_mult_2=None, fig1=None, fig2=None):
    """
    Plot multiplicity for two arrays. The first graph is just the first group of telescopes with the given divergence and the second is the second group of telescopes with their given divergence.
    And the last plot is a combiantion of both of them
    All the comments seen with the "#" is just to see that everything is working. The first part there is a comment from rad to deg because I had a mistake with it, but I am wokring on implementing the lines

    Parameters
    ----------
    array: Array object
        First array of telescopes
    array_2: Array object
        Second array of telescopes
    subarray_mult: array_like, optional
        Multiplicities for the telescopes (default is 1 for all)
        This is used for the subarrays
     subarray_mult_2: array_like, optional
        Multiplicities for the telescopes (default is 1 for all)
        This is used for the subarrays
    fig1: matplotlib.figure.Figure, optional
        First figure for array plot
    fig2: matplotlib.figure.Figure, optional
        Second figure for array_2 plot
    """
    if array.table.units == 'rad':
        array.__convert_units__(toDeg=True)
    
    if array_2.table.units == 'rad':
        array_2.__convert_units__(toDeg=True)
   
    # Get pointing coordinates for both arrays
    coord_1 = array.get_pointing_coord(icrs=False)
    #print(f"The coordinates are {coord_1}")
    coord_2 = array_2.get_pointing_coord(icrs=False)
    
    # Set Healpix resolution (nside)
    nside = 512
    map_multiplicity_1 = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    map_multiplicity_2 = np.zeros(hp.nside2npix(nside), dtype=np.float64)

    # Initialize Healpix coordinates
    counter = np.arange(0, hp.nside2npix(nside))
    ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
    coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    # Set multiplicities if not provided
    if subarray_mult_1 is None:
        subarray_mult_1 = np.ones(len(array.telescopes))
    if subarray_mult_2 is None:
        subarray_mult_2 = np.ones(len(array_2.telescopes))
   # total_number= len(array.telescopes) + len(array_2.telescopes)
   # Plotting for both arrays
    for i, tel in tqdm.tqdm(enumerate(array.telescopes)):
            # This is for array.telescopes
        #Trying to see if this works and then try to reput it in a more general way
            pointing = SkyCoord(ra=(coord_1.az[i].degree), dec=coord_1.alt[i].degree, unit='deg')
            #print(f"The az where it is pointing is: {coord_1.az[i].degree}")
            r_fov = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask = coordinate.separation(pointing) < r_fov
            map_multiplicity_1[mask] += subarray_mult_1[i]
            #print(map_multiplicity_1[mask])
    for i, tel in tqdm.tqdm(enumerate(array_2.telescopes)):
            # This is for array_2.telescopes
            #index_2 = i - len(array.telescopes)
            pointing_2 = SkyCoord(ra=coord_2.az[i].degree, dec=coord_2.alt[i].degree, unit='deg')
            r_fov_2 = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask_2 = coordinate.separation(pointing_2) < r_fov_2
            map_multiplicity_2[mask_2] += subarray_mult_2[i]
    #Trying to find the bigest R between the different configurations and taking that as universal
    R1 = np.sqrt(array.hFoV()[0] / np.pi) + 5
    print(R1)
    #print(array.table['fov'])
    #print(array_2.table['fov'])
    R2 = np.sqrt(array_2.hFoV()[0] / np.pi) + 5
    if R1>R2:
        R=R1
    else:
        R=R2 
    #The fist multiplicity plot
   
    hp.cartview(map_multiplicity_1, rot=[array.pointing["az"].value,
                                                      array.pointing["alt"].value],
                             lonra=[-R1, R1], latra=[-R1, R1],  cmap='viridis', nest=True,
                             return_projected_map=True, title=f"Map multiplicity 1 {array.div}")
    hp.graticule(dpar=5, dmer=5, coord='G', color='gray', lw=0.5)
    #print("The second map is")
    #The secod multiplicity plot 
    hp.cartview(map_multiplicity_2, rot=[array_2.pointing["az"].value,
                                                      array_2.pointing["alt"].value],
                         lonra=[-R2, R2], latra=[-R2, R2], cmap='viridis', nest=True,
                             return_projected_map=True, title=f"Map multiplicity 2 {array_2.div}")
    hp.graticule(dpar=5, dmer=5, coord='G', color='gray', lw=0.5)
    #The combination of both of them 
    hp.cartview(
        map_multiplicity_1+map_multiplicity_2,rot=[array.pointing["az"].value,
                                                      array.pointing["alt"].value],
                             lonra=[-R, R], latra=[-R, R],  cmap='viridis', nest=True,
                             return_projected_map=True, title=f"MapCombination1and2{array.div}and{array_2.div}")
    hp.graticule(dpar=5, dmer=5, coord='G', color='gray', lw=0.5)
    plt.show()
    
def multiplicity_plot_3_config(array, array_2, array_3, subarray_mult_1=None, subarray_mult_2=None, subarray_mult_3=None, fig1=None, fig2=None, fig3=None):
    """
    Plot multiplicity for three different configurations together. This could be helpful when we have MST and SST with two divergences. The first graph is just the first group of telescopes with the given divergence and the second is the second group of telescopes and the third is the third group of telescopes with their given divergence.
    And the last plot is a combiantion of both of them
    All the comments seen with the "#" is just to see that everything is working. The first part there is a comment from rad to deg because I had a mistake with it, but I am wokring on implementing the lines

    Parameters
    ----------
    array: Array object
        First array of telescopes
    array_2: Array object
        Second array of telescopes
    array_2: Array object
        Second array of telescopes
    subarray_mult: array_like, optional
        Multiplicities for the telescopes (default is 1 for all)
        This is used for the subarrays
     subarray_mult_2: array_like, optional
        Multiplicities for the telescopes (default is 1 for all)

    subarray_mult_3: array_like, optional
        Multiplicities for the telescopes (default is 1 for all)
        This is used for the subarrays
    fig1: matplotlib.figure.Figure, optional
        First figure for array plot
    fig2: matplotlib.figure.Figure, optional
        Second figure for array_2 plot
    """
    if array.table.units == 'rad':
        array.__convert_units__(toDeg=True)
    
    if array_2.table.units == 'rad':
        array_2.__convert_units__(toDeg=True)

    if array_3.table.units == 'rad':
        array_3.__convert_units__(toDeg=True)
   
    # Get pointing coordinates for both arrays
    coord_1 = array.get_pointing_coord(icrs=False)
    #print(f"The coordinates are {coord_1}")
    coord_2 = array_2.get_pointing_coord(icrs=False)

    coord_3 = array_3.get_pointing_coord(icrs=False)
    
    # Set Healpix resolution (nside)
    nside = 512
    map_multiplicity_1 = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    map_multiplicity_2 = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    map_multiplicity_3 = np.zeros(hp.nside2npix(nside), dtype=np.float64)

    # Initialize Healpix coordinates
    counter = np.arange(0, hp.nside2npix(nside))
    ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
    coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    # Set multiplicities if not provided
    if subarray_mult_1 is None:
        subarray_mult_1 = np.ones(len(array.telescopes))
    if subarray_mult_2 is None:
        subarray_mult_2 = np.ones(len(array_2.telescopes))
    if subarray_mult_3 is None:
        subarray_mult_3 = np.ones(len(array_3.telescopes))
   # total_number= len(array.telescopes) + len(array_2.telescopes)
   # Plotting for both arrays
    for i, tel in tqdm.tqdm(enumerate(array.telescopes)):
            # This is for array.telescopes
        #Trying to see if this works and then try to reput it in a more general way
            pointing = SkyCoord(ra=(coord_1.az[i].degree), dec=coord_1.alt[i].degree, unit='deg')
            #print(f"The az where it is pointing is: {coord_1.az[i].degree}")
            r_fov = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask = coordinate.separation(pointing) < r_fov
            map_multiplicity_1[mask] += subarray_mult_1[i]
            #print(map_multiplicity_1[mask])
    for i, tel in tqdm.tqdm(enumerate(array_2.telescopes)):
            # This is for array_2.telescopes
            #index_2 = i - len(array.telescopes)
            pointing_2 = SkyCoord(ra=coord_2.az[i].degree, dec=coord_2.alt[i].degree, unit='deg')
            r_fov_2 = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask_2 = coordinate.separation(pointing_2) < r_fov_2
            map_multiplicity_2[mask_2] += subarray_mult_2[i]

    for i, tel in tqdm.tqdm(enumerate(array_3.telescopes)):
            # This is for array_3.telescopes
            #index_3 = i - len(array.telescopes)
            pointing_3 = SkyCoord(ra=coord_3.az[i].degree, dec=coord_3.alt[i].degree, unit='deg')
            r_fov_3 = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask_3= coordinate.separation(pointing_3) < r_fov_3
            map_multiplicity_3[mask_3] += subarray_mult_3[i]
    #Trying to find the bigest R between the different configurations and taking that as universal
    R1 = np.sqrt(array.hFoV()[0] / np.pi) + 5
    print(R1)
    #print(array.table['fov'])
    #print(array_2.table['fov'])
    R2 = np.sqrt(array_2.hFoV()[0] / np.pi) + 5
    if R1>R2:
        R=R1
    else:
        R=R2 
    #The fist multiplicity plot
    R3 = np.sqrt(array_3.hFoV()[0] / np.pi) + 5
    hp.cartview(map_multiplicity_1, rot=[array.pointing["az"].value,
                                                      array.pointing["alt"].value],
                             lonra=[-R1, R1], latra=[-R1, R1],  cmap='viridis', nest=True,
                             return_projected_map=True, title=f"Map multiplicity 1 {array.div}")
    hp.graticule(dpar=5, dmer=5, coord='G', color='gray', lw=0.5)
    #print("The second map is")
    #The secod multiplicity plot 
    hp.cartview(map_multiplicity_2, rot=[array_2.pointing["az"].value,
                                                      array_2.pointing["alt"].value],
                         lonra=[-R2, R2], latra=[-R2, R2], cmap='viridis', nest=True,
                             return_projected_map=True, title=f"Map multiplicity 2 {array_2.div}")
    hp.graticule(dpar=5, dmer=5, coord='G', color='gray', lw=0.5)

    hp.cartview(map_multiplicity_3, rot=[array_3.pointing["az"].value,
                                                      array_3.pointing["alt"].value],
                         lonra=[-R3, R3], latra=[-R3, R3], cmap='viridis', nest=True,
                             return_projected_map=True, title=f"Map multiplicity 3 {array_3.div}")
    hp.graticule(dpar=5, dmer=5, coord='G', color='gray', lw=0.5)
    #The combination of both of them 
    hp.cartview(
        map_multiplicity_1+map_multiplicity_2+map_multiplicity_3,rot=[array.pointing["az"].value,
                                                      array.pointing["alt"].value],
                             lonra=[-R, R], latra=[-R, R],  cmap='viridis', nest=True,
                             return_projected_map=True, title=f"MapCombination 2 divergences SST: {array.div}and{array_2.div} MST: {array_3.div}")
    hp.graticule(dpar=5, dmer=5, coord='G', color='gray', lw=0.5)
    plt.show()
def multiplicity_plot(array, subarray_mult=None, fig=None):
        if array.table.units == 'rad':
            array.__convert_units__(toDeg=True)

        coord = array.get_pointing_coord(icrs=False)
        nside = 512
        map_multiplicity = np.zeros(hp.nside2npix(nside), dtype=np.float64)

        # Initialize Healpix coordinates
        counter = np.arange(0, hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
        coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        # If subarray_mult is not provided, set all multiplicities to 1
        if subarray_mult is None:
            subarray_mult = np.ones(len(array.telescopes))

        # Iterate over telescopes
        for i, tel in tqdm.tqdm(enumerate(array.telescopes)):
            pointing = SkyCoord(ra=coord.az[i].degree, dec=coord.alt[i].degree, unit='deg')
            r_fov = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask = coordinate.separation(pointing) < r_fov

            # Add intrinsic multiplicity for this telescope
            map_multiplicity[mask] += subarray_mult[i]
        
        R=np.sqrt(array.hFoV()[0]/np.pi) + 5
        hp.cartview(map_multiplicity, rot=[array.pointing["az"].value, array.pointing["alt"].value],
                lonra=[-R,R], latra=[-R,R], nest=True, cmap='viridis', title=f"{array.frame.site} div={array.div}")
        # Annotate with axis labels:
        plt.annotate('Right Ascension (degrees)', xy=(0.5, -0.05), xycoords='axes fraction', ha='center', va='center')
        plt.annotate('Declination (degrees)', 
                     xy=(-0.05, 0.5), xycoords='axes fraction', 
                     ha='center', va='center', rotation='vertical')
        hp.graticule(dpar=5, dmer=5, coord='G', color='gray', lw=0.5)

        plt.show()


def table_multiplicity(array, subarray_mult=None, maximum_multiplicity=None, step=None, fig=None):
    """
       Make the table of the FoV and the multiplicity so we can see the part of the multiplicity that is going for the FoV. The question is how to do it.
       So basically we are using a part of the hFoV and in the hFoV we could pass it the m_cut so maybe we could do that here, so we get the image and we calculate the hFoV with the m_cut of a certain degree everytime for example we could do 14,13,12,and so on up to 0

        Parameters
        ----------
        array: Array object
            First array of telescopes
        subarray_mult: array_like, optional
            Multiplicities for the telescopes (default is 1 for all)
            This is used for the subarrays
    
            fig1: matplotlib.figure.Figure, optional
            First figure for array plot
   
        """
    if array.table.units == 'rad':
        array.__convert_units__(toDeg=True)
    if maximum_multiplicity is None:
        maximum_multiplicity=13#Actually I could caluclate this
    if step is None:
        step=1
    hFoV_array=[]
    multiplicities = list(range(1, maximum_multiplicity+1))
    for i in range(maximum_multiplicity):
       # print(i+1)
        hFoV_array.append(array.hFoV(subarray_mult=subarray_mult,m_cut=i+1)[0])
       # print(hFoV_array)
    plt.figure(figsize=(8, 6)) 
    plt.bar(multiplicities, hFoV_array, color='limegreen')
    plt.xlabel("Multiplicity greater than the number")
    plt.ylabel("Hyper Field of View (hFoV)")
    plt.title("hFoV vs. Multiplicity greater than the number")
    custom_labels = [f'> {m}' for m in multiplicities]
    plt.xticks(multiplicities, custom_labels)
    plt.grid(axis='y', alpha=0.4)
    plt.show()
    

def combination_bar_graph(array, array_2, subarray_mult=None,subarray_mult_2=None, maximum_multiplicity=None, step=None, fig=None):
    if array.table.units == 'rad':
        array.__convert_units__(toDeg=True)
    if array_2.table.units == 'rad':
        array_2.__convert_units__(toDeg=True)
    if maximum_multiplicity is None:
        maximum_multiplicity=13#Actually I could caluclate this
    if step is None:
        step=1
    hFoV_array=[]
    hFoV_array_2=[]
    multiplicities = list(range(1, maximum_multiplicity+1))
    multiplicities_2= list(range(1, maximum_multiplicity+1))
    for i in range(maximum_multiplicity):
       # print(i+1)
        hFoV_array.append(array.hFoV(subarray_mult=subarray_mult,m_cut=i+1)[0])
        hFoV_array_2.append(array.hFoV(subarray_mult=subarray_mult_2,m_cut=i+1)[0])
       # print(hFoV_array)
    plt.figure(figsize=(8, 6)) 
    plt.bar(multiplicities, hFoV_array, color='darkmagenta',alpha=0.4)
    plt.bar(multiplicities_2, hFoV_array_2, color='darkgreen',alpha=0.4)
    plt.xlabel("Multiplicity greater than the number")
    plt.ylabel("Hyper Field of View (hFoV)")
    plt.title("hFoV vs. Multiplicity greater than the number confronting 2 configurations")
    custom_labels = [f'> {m}' for m in multiplicities]
    plt.xticks(multiplicities, custom_labels)
    plt.grid(axis='y', alpha=0.4)
    plt.show()


def combination_bar_graph_av_mult(array, array_2, subarray_mult=None,subarray_mult_2=None, maximum_multiplicity=None, step=None, fig=None):
    if array.table.units == 'rad':
        array.__convert_units__(toDeg=True)
    if array_2.table.units == 'rad':
        array_2.__convert_units__(toDeg=True)
    if maximum_multiplicity is None:
        maximum_multiplicity=13#Actually I could caluclate this
    if step is None:
        step=1
    hFoV_array=[]
    hFoV_array_2=[]
    av_mult_array=[]
    av_mult_array_2=[]
    multiplicities = list(range(1, maximum_multiplicity+1))
    multiplicities_2= list(range(1, maximum_multiplicity+1))
    graph_mult=list(range(5, maximum_multiplicity+1))
    for i in range(maximum_multiplicity):
       # print(i+1)
        av_mult_array.append(array.hFoV(subarray_mult=subarray_mult,m_cut=i+1)[1])
        av_mult_array_2.append(array.hFoV(subarray_mult=subarray_mult_2,m_cut=i+1)[1])
        hFoV_array.append(array.hFoV(subarray_mult=subarray_mult,m_cut=i+1)[0])
        hFoV_array_2.append(array.hFoV(subarray_mult=subarray_mult_2,m_cut=i+1)[0])
       # print(hFoV_array)
    plt.figure(figsize=(8, 6)) 
    plt.scatter(av_mult_array, hFoV_array, color='darkmagenta', alpha=0.6, label='Config 1', marker='o', s=100)
    plt.scatter(av_mult_array_2, hFoV_array_2, color='darkgreen', alpha=0.6, label='Config 2', marker='s', s=40)
    texts = []
    for i, (x, y) in enumerate(zip(av_mult_array, hFoV_array)):
        texts.append(plt.text(x, y, f"{i+1}", fontsize=12, ha="right", va="center", color='darkmagenta'))
    
    for i, (x, y) in enumerate(zip(av_mult_array_2, hFoV_array_2)):
        texts.append(plt.text(x, y, f"{i+1}", fontsize=12, ha="left", va="center", color='darkgreen'))
    
    adjust_text(texts, only_move={'points':'y', 'text':'y'}) # in case we want arrows: ,arrowprops=dict(arrowstyle='-', color='gray')
    plt.ylabel("hFoV")
    plt.xlabel("Average Multiplicity")
    plt.title("Av Multiplicity vs. hFoV for different values of m_cut")
    plt.legend()
    plt.grid(axis='y', alpha=0.1)
    plt.show()
    
def multiplicity_plot_old(array, fig=None):
   
    nside = 512
    map_multiplicity = np.zeros(hp.nside2npix(nside), dtype=np.int8)
    counter = np.arange(0, hp.nside2npix(nside))
    ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
    coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    coord = array.get_pointing_coord(icrs=False)
    for i in tqdm.tqdm(range(len(array.telescopes))):
        pointing = SkyCoord(ra=coord.az[i].degree, dec=coord.alt[i].degree, unit='deg')
        r_fov = np.arctan((array.telescopes[i].camera_radius/array.telescopes[i].focal).to(u.dimensionless_unscaled)).to(u.deg)
        mask = coordinate.separation(pointing) < r_fov
        map_multiplicity[mask] += 1
    
    
    R=np.sqrt(array.hFoV()[0]/np.pi) + 5
    hp.cartview(map_multiplicity, rot=[array.pointing["az"].value, array.pointing["alt"].value],
                lonra=[-R,R], latra=[-R,R], nest=True, cmap='viridis', title=f"{array.frame.site} div={array.div}")
    # Annotate with axis labels:
    plt.annotate('Right Ascension (degrees)', xy=(0.5, -0.05), xycoords='axes fraction', ha='center', va='center')
    plt.annotate('Declination (degrees)', 
                 xy=(-0.05, 0.5), xycoords='axes fraction', 
                 ha='center', va='center', rotation='vertical')
    hp.graticule(dpar=5, dmer=5, coord='G', color='gray', lw=0.5)
    
    plt.show()

