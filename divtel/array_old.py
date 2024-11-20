import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import ipywidgets as widgets

from astropy.table import Table
import astropy.table as tab

from . import utils as utils
from . import visualization as visual
from . import pointing

from .cta import CTA_Info

from astropy.coordinates import SkyCoord

from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, Point

import copy 

    
class Array:

    """
    Class containing information on array (a set of telescopes)
    
    Parameters
    ----------
    telescope_list: array, class.Telescope
        Array containing class.Telescope
    frame: class.CTA_Info
        CTA Information
    kwargs: dict
        args for class.CTA_Info
    """

    def __init__(self, telescope_list, frame=None, pointing2src=False, **kwargs):
        
        self.telescopes = telescope_list

        self._div = 0
        self._pointing = {"az":0*u.deg, "alt":0*u.deg, "ra": 0*u.deg, "dec": 0*u.deg}

        if frame == None:
            self._frame = CTA_Info(verbose=False, **kwargs)
        else:
            self._frame = copy.copy(frame)
            if pointing2src and (self.frame.source is not None):
                self.set_pointing_coord(ra = self.frame.source.icrs.ra.deg, 
                                        dec = self.frame.source.icrs.dec.deg)

        self.__make_table__()

    def __make_table__(self):

        """
        Merge rows from Telescope.table

        Returns
        -------
        astropy.table
        """
        
        table = []

        for tel in self.telescopes:

            table.append(tel.table)
        
        units = 'rad'
        if hasattr(self, "_table"):
            if hasattr(self._table, "units"):
                units = self.table.units
        
        self._table = tab.vstack(table)

        self._table.add_column(self._dist2tel(), name="d_tel")
        self._table["d_tel"].unit = u.m
        self._table["d_tel"].info.format = "{:.2f}"

        self._table.units = units

    def __convert_units__(self, toDeg=True):
        """
        Convert units in the table from deg to rad, and vice versa.
    
        Parameters
        ----------
        toDeg: bool, optional
        """

        self._table = utils.deg2rad(self._table, toDeg)
        if toDeg:
            self._table.units = 'deg'
        else:
            self._table.units = 'rad'

    def _dist2tel(self):
        """
        Distance to the telescope from the barycenter
    
        Returns
        -------
        float
        """

        dist = np.zeros(self.size_of_array)
        for i, axis in enumerate(["x", "y", "z"]):
            dist += (self.table[axis] - self.barycenter[i])**2.
        dist = np.sqrt(dist)
        return dist

    @property
    def table(self):
        """
        Return astropy table containing all telescope information
        An attribute, Array.table.units, determines the units of `az`, `alt`, `zn`, 
        `radius`, `fov` units (deg or rad)

        Returns
        -------
        astropy.table
        """
        if hasattr(self._table, "units"):
            if (self._table.units == 'deg')*(self._table["az"].unit == u.rad):
                self._table = utils.deg2rad(self._table, True)
            elif (self._table.units == 'rad')*(self._table["az"].unit == u.deg):
                self._table = utils.deg2rad(self._table, False)
        return self._table

    @property
    def size_of_array(self):
        """
        Return the number of telescopes

        Returns
        -------
        float
        """
        return self._table.__len__()

    @property
    def frame(self):
        """
        Return a frame

        Returns
        -------
        class.CTA_info
        """
        return self._frame
    
    @property
    def barycenter(self):
        """
        Return a barycenter

        Returns
        -------
        array
            [b_x, b_y, b_z]
        """
        return np.array(utils.calc_mean(self.table, ["x", "y", "z"]))

    @property
    def div(self):
        """
        Return a divergence parameter

        Returns
        -------
        float
        """
        return self._div

    @property
    def pointing(self):
        """
        Return pointing information

        Returns
        -------
        dict
            keys: `ra`, `dec`, `az`, and `alt`
        """
        return self._pointing

    def calc_mean(self, params):
        """
        Calculate the mean values of parameters
        
        Parameters
        ----------
        params: str or array
            see ArrayConfig.utils.calc_mean

        Returns
        -------
        array
        """
        return np.array(utils.calc_mean(self.table, params))

    def hFoV(self, m_cut=0, return_multiplicity=False, full_output=False):
        """
        Return a hyper field of view (hFoV) above a given multiplicity.
    
        Parameters
        ----------
        m_cut: float, optional
            the minimum multiplicity
        return_multiplicity: bool, optional
            return average and variance of multiplicity
        full_output: bool, optional
            return all parameters; multiplicity, overlaps, geoms
        Returns
        -------
        fov: float
            hFoV
        m_ave: float
            average of multiplicity
        m_var: float
            variance of multiplicity
        multiplicity: array
            array containing multiplicity and corresponding hFoV
        overlaps: array
            array containing the number of overlaps for each patch
        geoms: shapely.ops.polygonize
            geometry of each patch
        """
        if self.table.units == 'rad':
            self.__convert_units__(toDeg=True)
        
        coord = self.get_pointing_coord(icrs=False)
        
        if max(self.table["az"])-min(self.table["az"]) > 180:
            polygons = []
            for az, alt, r in zip(coord.az.degree, coord.alt.degree, self.table["radius"]):
                if az < 180:
                    polygons.append(Point(az, alt).buffer(r))
                else:
                    polygons.append(Point(az-360, alt).buffer(r))
        else:
            polygons = [Point(az, alt).buffer(r) for az, alt, r in zip(coord.az.degree, coord.alt.degree, self.table["radius"])]

        union = unary_union([LineString(list(pol.exterior.coords)) for pol in polygons])
        geoms = [geom for geom in polygonize(union)]
        hfov = [geom.area for geom in geoms]

        count_overlaps = np.zeros(len(geoms))
        for i, geom in enumerate(geoms):
            count_overlaps[i] = sum([1 for pol in polygons if abs(geom.difference(pol).area)<1e-5])

        hfov = np.array(hfov)

        # multiplicity associated with each patch
        overlaps = np.array(count_overlaps)
        eff_overlaps=[]
        eff_geoms=[]
        for i in range(len(overlaps)):
            if overlaps[i]>m_cut:
                eff_overlaps.append(overlaps[i])
                eff_geoms.append(geoms[i])
        multiplicity = np.array([[i, hfov[overlaps==i].sum()] for i in set(overlaps)])
        
        fov = sum(multiplicity[:,1][multiplicity[:,0]>=m_cut])*u.deg**2

        if full_output:
            return multiplicity, eff_overlaps, eff_geoms
        elif return_multiplicity:
            m_ave = np.average(multiplicity[:,0], weights=multiplicity[:,1])
            m_var = np.average((multiplicity[:,0]-m_ave)**2, weights=multiplicity[:,1])
            return fov, m_ave, m_var
        else:
            return fov
            
    
    def update_frame(self, site=None, time=None, delta_t=None, verbose=False):
        """
        Update class.CTA_Info parameters (site and/or observation time)
    
        Parameters
        ----------
        site: str, optional
            Updated site name
        time: str, optional 
            Updated observation time (yyyy-MM-ddThh:mm:ss)
        delta_t: astropy.Quantity, optional 
            Elapsed time from the original observation time.
            e.g., CTA_Info.update(delta_t= -0.5*u.hour) 
            -> t_new = t_old - 0.5 hour
        verbose: optional 
        """

        self.frame.update(site=site, time=time, delta_t=delta_t, verbose=verbose)
        self.divergent_pointing(self.div, ra=self.pointing["ra"], dec=self.pointing["dec"])
    
    def get_pointing_coord(self, icrs=True):
        """
        Return pointing coordinates
    
        Parameters
        ----------
        icrs: bool, optional
            If True, return (ra, dec). 
            If False, return (alt, az)
        """
        return pointing.pointing_coord(self.table, self.frame, icrs=icrs)

    def set_pointing_coord(self, src=None, ra = None, dec = None, alt=None, az = None, units='deg'):
        """
        Set pointing coordinates
    
        Parameters
        ----------
        src: astropy.coordinates.SkyCoord, optional
        ra: float, optional
            Right ascension of a source
        dec: float, optional
            Declination of a source
        alt: float, optional
            Mean altitude of the array
        az: float, optional
            Mean azumith angle of the array
        units: str
            Units of RA and DEC; either deg (default) or rad
        """
        if type(src) == SkyCoord:
            ra = src.icrs.ra.deg
            dec = src.icrs.dec.deg
            units = 'deg'
            
        if ra is not None and dec is not None:
            src = self.frame.set_source_loc(ra=ra, dec=dec, units=units)
            self._pointing["ra"] = src.icrs.ra.value * u.deg
            self._pointing["dec"] = src.icrs.dec.value * u.deg
            self._pointing["alt"] = src.alt.value * u.deg
            self._pointing["az"] = src.az.value * u.deg
        elif alt is not None and az is not None:
            if units == "deg":
                self._pointing["alt"] = alt * u.deg
                self._pointing["az"] = az * u.deg
            else:
                self._pointing["alt"] = alt * u.rad
                self._pointing["alt"] = az * u.rad
        
        for tel in self.telescopes:
            tel.__point_to_altaz__(self.pointing["alt"], self.pointing["az"])

        self.__make_table__()
        
    def divergent_pointing(self, div, ra=None, dec = None, alt=None, az=None, units="deg"):
        """
        Divergent pointing given a parameter div.
        Update pointing of all telescopes of the array.

        Parameters
        ----------
        div: float between 0 and 1
        ra: float, optioanl
            source ra 
        dec: float, optional
            source dec 
        alt: float, optional
            mean alt pointing
        az: float, optional
            mean az pointing
        units: string, optional
            either 'deg' (default) or 'rad'
        """

        self.set_pointing_coord(ra=ra, dec = dec, alt=alt, az=az, units=units)

        self._div = div
        
        if np.abs(div) > 1: #or div < 0:
            print("[Error] The div abs value should be lower and 1.")
        elif div!=0:
            G = pointing.pointG_position(self.barycenter, self.div, self.pointing["alt"], self.pointing["az"])
            for tel in self.telescopes:
                alt_tel, az_tel = pointing.tel_div_pointing(tel.position, G)
                
                if div < 0:
                    az_tel=az_tel - np.pi 
                	
                tel.__point_to_altaz__(alt_tel*u.rad, az_tel*u.rad)
                
        
            self.__make_table__()
    
    def group_by(self, group = None):
        
        if type(group) == dict:
            groupping = np.zeros(self.size_of_array)
            labels = []
            j = 1
            for key in group.keys():
                labels.append(key)
                for i in group[key]:
                    groupping[i-1] = j
                j+=1
            tel_group = self._table.group_by(np.asarray(groupping))
        elif group:
            tel_group = self._table.group_by("radius")
            labels = ["group_{}".format(i+1) for i in range(len(tel_group.groups))]
        else:
            tel_group = self._table.group_by(np.zeros(self.size_of_array))
            labels = ["_nolegend_"]
        return (tel_group, labels)

 
    def display(self, projection, ax=None, group=False, **kwargs):
        """
        Display the CTA array

        Parameters
        ----------
        projection: str
            any combination of `x`, `y`, and `z` or `skymap`
        ax: pyplot.axes, optional
        group: bool or dict, optional
        kwargs: args for `pyplot.scatter`

        Returns
        -------
        ax: `matplotlib.pyplot.axes`
        """
        tel_group, labels = self.group_by(group)

        if projection == 'skymap':
            for i, [table, label] in enumerate(zip(tel_group.groups, labels)):
            
                ax = visual.display_skymap(table, self.frame,  
                                    label=labels[i], ax=ax)
            return ax
        else:
            proj = []
            for axis in ["x", "y", "z"]:
                if axis in projection:
                    proj.append(axis)

        if len(proj) == 1:
            ax = visual.display_1d(tel_group, proj, ax=ax, labels=labels, **kwargs)
        elif len(proj) == 2:
            ax = visual.display_2d(tel_group, proj, ax=ax, labels=labels, **kwargs)
        else:
            ax = visual.display_3d(tel_group, proj, ax=ax, labels=labels, **kwargs)
        
        return ax

    def skymap_polar(self, group=None, fig=None, filename=None):
        """
        Plot skymap

        Parameters
        ----------
        group: bool or dict, optional
        fig: pyplot.figure, optional
        filemane: str, optional
        
        """
        return visual.skymap_polar(self, group=group, fig=fig, filename=filename)

    def multiplicity_plot(self,fig=None):
        """
        Plot multiplicity

        Parameters
        ----------
        fig: pyplot.figure, optional
        """
        return visual.multiplicity_plot(self, fig=fig)

    def export_cfg(self, filename=None, outdir="./", verbose=False):
        """
        Export cfg file.

        Parameters
        ----------
        filename: str, optional
            A default name is 'CTA-ULTRA6-LaPalma-divX-azX-altX.cfg'

        outdir: str, optional,
        
        verbose: bool, optional
        """

        if filename==None:
            filename = 'CTA-ULTRA6-LaPalma-div{}-az{}-alt{}.cfg'.format(
                str(self.div).replace(".", "_"), 
                str(self.pointing["az"].value).replace(".", "_"), 
                str(self.pointing["alt"].value).replace(".", "_"))
        
        with open(outdir+filename, 'w') as f:
            f.write('#ifndef TELESCOPE\n')
            f.write('#  define TELESCOPE 0\n')
            f.write('#endif\n')
            f.write('#if TELESCOPE == 0\n')
            f.write('   TELESCOPE_THETA={:.2f} \n'.format(90 - self.pointing["alt"].value))
            f.write('   TELESCOPE_PHI={:.2f} \n'.format(self.pointing["az"].value))
            f.write('\n% Global and default configuration for things missing in telescope-specific config.\n')
            f.write('#  include <CTA-ULTRA6-LST.cfg>\n')    
            for n, tel in enumerate(self.table, 1):
                zd = 90 - tel['alt']
                f.write('\n#elif TELESCOPE == {:d}\n'.format(n))
                if n <= 4:
                    f.write('#  include <CTA-ULTRA6-LST.cfg>\n')
                else:
                    f.write('#  include <CTA-ULTRA6-MST-NectarCam.cfg>\n')
                f.write('   TELESCOPE_THETA={:.2f}\n'.format(zd))
                f.write('   TELESCOPE_PHI={:.2f}\n'.format(360 - tel["az"]))
            f.write('#else\n')
            f.write('   Error Invalid telescope for CTA-ULTRA6 La Palma configuration.\n')
            f.write('#endif\n')
            f.write('trigger_telescopes = 2 % In contrast to Prod-3 South we apply loose stereo trigger immediately\n')
            f.write('array_trigger = array_trigger_ultra6_diver-test.dat\n')
        
        if verbose:
            with open(outdir+filename, 'r') as f:
                for line in f.readlines():
                    print(line)

