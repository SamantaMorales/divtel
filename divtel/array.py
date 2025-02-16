import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import ipywidgets as widgets

from astropy.table import Table
import astropy.table as tab

from . import utils as utils
from . import visualization as visual
from . import pointing
from adjustText import adjust_text
from .cta import CTA_Info

from astropy.coordinates import SkyCoord

#from shapely.ops import unary_union, polygonize
#from shapely.geometry import LineString, Point

import copy 

import healpy as hp
import tqdm
    
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

    def hFoV_old(self, m_cut=0, return_multiplicity=False):
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
        nside = 512
        map_multiplicity = np.zeros(hp.nside2npix(nside), dtype=np.int8)
        counter = np.arange(0, hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
        coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        for i in tqdm.tqdm(range(len(self.telescopes))):
            pointing = SkyCoord(ra=coord.az[i].degree, dec= coord.alt[i].degree, unit='deg')
            r_fov = np.arctan((self.telescopes[i].camera_radius/self.telescopes[i].focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask = coordinate.separation(pointing) < r_fov
            map_multiplicity[mask] += 1

       
        mask_fov = map_multiplicity> m_cut
        #mask_fov_eff = map_multiplicity>3

        hfov = hp.nside2pixarea(nside, True)*np.sum(mask_fov)
        m_ave = np.mean(map_multiplicity[mask_fov])
        return hfov, m_ave     

    
    def hFoV(self, m_cut=0, return_multiplicity=False, subarray_mult=None):
        """   
        Return a hyper field of view (hFoV) above a given multiplicity with optional subgroup handling.
    
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
        # Unit conversion remains the same
        if self.table.units == 'rad':
            self.__convert_units__(toDeg=True)

        coord = self.get_pointing_coord(icrs=False)
        nside = 512
        map_multiplicity = np.zeros(hp.nside2npix(nside), dtype=np.float64)

        # Initialize Healpix coordinates
        counter = np.arange(0, hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
        coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        # If subarray_mult is not provided, set all multiplicities to 1
        if subarray_mult is None:
            subarray_mult = np.ones(len(self.telescopes))

        # Iterate over telescopes
        for i, tel in tqdm.tqdm(enumerate(self.telescopes)):
            pointing = SkyCoord(ra=coord.az[i].degree, dec=coord.alt[i].degree, unit='deg')
            r_fov = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask = coordinate.separation(pointing) < r_fov

            # Add intrinsic multiplicity for this telescope
            map_multiplicity[mask] += subarray_mult[i]

        # Calculate the hFoV and average multiplicity
        mask_fov = map_multiplicity > m_cut
        hfov = hp.nside2pixarea(nside, True) * np.sum(mask_fov)
        m_ave = np.mean(map_multiplicity[mask_fov])

        return hfov, m_ave



    def hFoV_for_2_arrays(self, array_2, m_cut=0, return_multiplicity=False, subarray_mult=None, subarray_mult_2=None):
        """   
        Return a hyper field of view (hFoV) above a given multiplicity with optional subgroup handling.
        Here the idea is to do the same thing as before, getting two arrays and using that
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
        
        # Unit conversion remains the same
        if self.table.units == 'rad':
            self.__convert_units__(toDeg=True)

        if array_2.table.units == 'rad':
            array_2.__convert_units__(toDeg=True)

        coord = self.get_pointing_coord(icrs=False)
       # print(coord)
        coord_2 = array_2.get_pointing_coord(icrs=False)
       # print(f"the coord_2 are: {coord_2}")
        nside = 512
        map_multiplicity_1 = np.zeros(hp.nside2npix(nside), dtype=np.float64)
        map_multiplicity_2= np.zeros(hp.nside2npix(nside), dtype=np.float64)

        # Initialize Healpix coordinates
        counter = np.arange(0, hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
        coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        #Inirialize Healpix coordinates for second array, just becausae I want to make sure everything is 
        #working
        counter_2 = np.arange(0, hp.nside2npix(nside))
        ra_2, dec_2 = hp.pix2ang(nside, counter_2, True, lonlat=True)
        coordinate_2 = SkyCoord(ra=ra_2*u.deg, dec=dec_2*u.deg)

        # If subarray_mult is not provided, set all multiplicities to 1
        if subarray_mult is None:
            subarray_mult = np.ones(len(self.telescopes))

        # If subarray_mult is not provided, set all multiplicities to 1
        if subarray_mult_2 is None:
            subarray_mult_2 = np.ones(len(array_2.telescopes))

        # Iterate over telescopes
        for i, tel in tqdm.tqdm(enumerate(self.telescopes)):
            pointing = SkyCoord(ra=coord.az[i].degree, dec=coord.alt[i].degree, unit='deg')
            r_fov = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask = coordinate.separation(pointing) < r_fov
            # Add intrinsic multiplicity for this telescope
            map_multiplicity_1[mask] += subarray_mult[i]

         # Iterate over telescopes
        for i, tel in tqdm.tqdm(enumerate(array_2.telescopes)):
            pointing_2 = SkyCoord(ra=coord_2.az[i].degree, dec=coord_2.alt[i].degree, unit='deg')
            r_fov_2 = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask_2 = coordinate_2.separation(pointing_2) < r_fov_2
            # Add intrinsic multiplicity for this telescope
            map_multiplicity_2[mask_2] += subarray_mult_2[i]
            
        # Calculate the hFoV and average multiplicity
        combination_map=map_multiplicity_1 + map_multiplicity_2
        mask_fov = map_multiplicity_1 + map_multiplicity_2 > m_cut #Here I am adding both of them so now try
        hfov = hp.nside2pixarea(nside, True) * np.sum(mask_fov)
        m_ave = np.mean(combination_map[mask_fov])

        return hfov, m_ave

    def combiantion_of_FoV( self,number_of_arrays=None, array_2=None, array_3=None, array_4=None, subarray_mult_1=None, subarray_mult_2=None, subarray_mult_3=None, subarray_mult_4=None, m_cut=0):
        array_1=self
        if number_of_arrays is None:
            number_of_arrays=1
        nside = 512
        map_multiplicities=[]
        subarray_different_multiplicities=[]
        different_arrays=[]
        map_multiplicity_list = [np.zeros(hp.nside2npix(nside), dtype=np.float64) for _ in range(number_of_arrays)]
        coord_list=[]
        if subarray_mult_1 is None:
                subarray_mult_1 = np.ones(len(self.telescopes))
        if subarray_mult_2 is None:
                subarray_mult_2 = np.ones(len(array_2.telescopes))
        if subarray_mult_3 is None:
                subarray_mult_3 = np.ones(len(array_3.telescopes))
        if array_4 is not None and subarray_mult_4 is None:
                subarray_mult_4 = np.ones(len(array_4.telescopes))
        combination_map=np.zeros(hp.nside2npix(nside), dtype=np.float64)
        subarray_different_multiplicities = [x for x in [subarray_mult_1, subarray_mult_2, subarray_mult_3, subarray_mult_4] if x is not None]
        different_arrays=[array_1, array_2, array_3, array_4]
        for number_array in range(number_of_arrays):
          #  print(number_array)
            coord_list.append(different_arrays[number_array].get_pointing_coord(icrs=False))
            map_multiplicities.append(np.zeros(hp.nside2npix(nside), dtype=np.float64))
            counter = np.arange(0, hp.nside2npix(nside))
            ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
            coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            # Iterate over telescopes
            for i, tel in tqdm.tqdm(enumerate(different_arrays[number_array].telescopes)):
               # print(f"The az{coord_list[number_array].az[i].degree}")
                pointing = SkyCoord(ra=coord_list[number_array].az[i].degree, dec=coord_list[number_array].alt[i].degree, unit='deg')
                r_fov = np.arctan((tel.camera_radius / tel.focal).to(u.dimensionless_unscaled)).to(u.deg)
                mask = coordinate.separation(pointing) < r_fov
               # print(subarray_different_multiplicities[number_array][i])
                #print( map_multiplicity_list[number_array][mask])
              #  print(f"the diff_mult{subarray_different_multiplicities[number_array][i]}")
                map_multiplicity_list[number_array][mask] += subarray_different_multiplicities[number_array][i]
        
        
    # Sum all subarray maps into the combination map
        for i in range(number_of_arrays):
           # print(map_multiplicities[i])
            combination_map += map_multiplicity_list[i]
            mask_fov = combination_map > m_cut #Here I am adding both of them so now try
            hfov = hp.nside2pixarea(nside, True) * np.sum(mask_fov)
           # print(combination_map)
           # print(mask_fov)
            m_ave = np.mean(combination_map[mask_fov])

        return hfov, m_ave

            

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

        self.set_pointing_coord(ra=ra, dec = dec, alt=alt, az=az, units=units) #repointing
        self._div = div
        
        if np.abs(div) > 1: #or div < 0:
            print("[Error] The div abs value should be lower and 1.")
        elif div!=0:
            G = pointing.pointG_position(self.barycenter, self.div, self.pointing["alt"], self.pointing["az"])
            for tel in self.telescopes:
                alt_tel, az_tel = pointing.tel_div_pointing(tel.position, G)
                
                if div < 0:
                    az_tel=az_tel - np.pi 
                    radians= az_tel * u.rad
                    print(radians)
                	
                tel.__point_to_altaz__(alt_tel*u.rad, az_tel*u.rad)
                
        
            self.__make_table__()


    def divergent_pointing_2_div(self, tel_group_2, complete_array, div1, div2, ra=None, dec = None, alt=None, az=None, units="deg"):
        """
        Divergent pointing given two parameters div1 and div2.
        The idea is to use a 2 divergence to have two telecopes and not only one. So I am giving it two different sets of telescopes in group 1 and         group 2. The self stays because I need it... maybe though we should think about this! Well in the case we have two different ones it means  
        that we will have 2 different telescopes groups but maybe it makes sense.... so the question is how to work with these? Do we slice the 
        array? And how to slice them giving them a limit? hmmmm these changes things But the idea, maybe, it would be to just use them separetly. 
        This would be interesting as well to do the visulization and all for a MST and SST toegther later on to plot them one above an other..... 
        Hmmmmmmmm but in that case.... hmmmmmmmmmmmmmmm..... as well in the multiplicity plot. Okay so or either split them in a sense here or 
        split them in the program and than give them as an array. For the beginning to see if it works I will split them in the program and give it 
        here just the arrays. An other question is how can we put together different telescopes
         Update pointing of all telescopes of the array.

        Parameters
        ----------
        div1: float between 0 and 1
        div1: float between 0 and 1
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
        
        self._div = div1

        tel_group_2.set_pointing_coord(ra=ra, dec = dec, alt=alt, az=az, units=units)

        tel_group_2._div = div2
        
        if np.abs(div1) > 1: #or div < 0:
            print("[Error] The div abs value should be lower and 1.")
        if np.abs(div2) > 1: #or div < 0:
            print("[Error] The div abs value should be lower and 1.")    
        elif div1 != 0 and div2 != 0:
            #TO put the same barycenter maybe I could just change here instead of self.barycenter put the barycenter of the full_array, ask confirmation. Now at this point I should change the display as well
            G1 = pointing.pointG_position(complete_array.barycenter, self.div, self.pointing["alt"], self.pointing["az"])
            print(f"the barycenter for the calculations{complete_array.barycenter}")
            G2 = pointing.pointG_position(complete_array.barycenter, tel_group_2.div, tel_group_2.pointing["alt"], tel_group_2.pointing["az"])
            print(f"the barycenter for the calculations of the second divergence{complete_array.barycenter}")
            for tel_1 in self.telescopes:
                alt_tel_1, az_tel_1 = pointing.tel_div_pointing(tel_1.position, G1)
               # print(f"the azimuth of tel 1 is{az_tel_1}")
                if div1 < 0:
                    az_tel_1=az_tel_1 - np.pi 
                 #   print(f"It was negative and the az_tel_1 is: {az_tel_1}")
                tel_1.__point_to_altaz__(alt_tel_1*u.rad, az_tel_1*u.rad)
                #print(f"The azimuth is {tel_1.__point_to_altaz__(alt_tel_1*u.rad, az_tel_1*u.rad)}")
            for tel_2 in tel_group_2.telescopes:
                alt_tel_2, az_tel_2 = pointing.tel_div_pointing(tel_2.position, G2)
                
                #print(f"the azimuth of tel_2 is{az_tel_2}")
                
                if div2< 0:
                    
                    az_tel_2=az_tel_2 - np.pi
                    #print(f"divergence negative(convergence) and the azimuth of telescope 2 is actually is{az_tel_2}")
                tel_2.__point_to_altaz__(alt_tel_2*u.rad, az_tel_2*u.rad)
                
                

            self.__make_table__()
            tel_group_2.__make_table__()
                  
    
    
    def group_by(self, group = None):
        number_of_telescopes_subarray=[]
        if type(group) == dict:
            groupping = np.zeros(self.size_of_array)
            labels = []
            number_of_telescopes_subarray=[]
            j = 1
            for key, indices in group.items():
                labels.append(key) #labels are like 1,2,3,4
                index_of_sub_telescope= 0 #we start it at 0
                for idx in indices: #here I am working witht the ones inside 
                    index_of_sub_telescope += 1 
                number_of_telescopes_subarray.append(index_of_sub_telescope)
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
        return (tel_group, labels, number_of_telescopes_subarray)
    

 
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
        tel_group, labels, number_of_telescopes_subarray = self.group_by(group)

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

    def multiplicity_plot_2_div(self, array_2, subarray_mult_1=None, subarray_mult_2=None, fig1=None, fig2=None):
        """
        Plot multiplicity

        Parameters
        ----------
        fig: pyplot.figure, optional
        """
            
        return visual.multiplicity_plot_2_div(self, array_2, subarray_mult_1, subarray_mult_2, fig1=fig1, fig2=fig2)

    def multiplicity_plot_3_config(self, array_2,array_3, subarray_mult_1=None, subarray_mult_2=None, subarray_mult_3=None,fig1=None, fig2=None, fig3=None):
        """
        Plot multiplicity

        Parameters
        ----------
        fig: pyplot.figure, optional
        """
            
        return visual.multiplicity_plot_3_config(self, array_2, array_3, subarray_mult_1, subarray_mult_2, subarray_mult_3, fig1=fig1, fig2=fig2, fig3=fig3)

    def table_multiplicity(self, subarray_mult=None, maximum_multiplicity=None, step=None, fig=None):
        """
        bar graph of mult and hFov
        Parameters
        ----------
        fig: pyplot.hist, optional
        """

        return visual.table_multiplicity(self, subarray_mult, fig=fig)

    def combination_bar_graph(self, array_2, subarray_mult=None,subarray_mult_2=None, maximum_multiplicity=None, step=None, fig=None):

         """
        bar graph mult and hFov of 2 arrat
        Parameters
        ----------
        fig: pyplot.hist, optional
        """

         return visual.combination_bar_graph(self, array_2, subarray_mult, subarray_mult_2, fig=fig)
    def combination_bar_graph_av_mult(self, array_2, subarray_mult=None,subarray_mult_2=None, maximum_multiplicity=None, step=None, fig=None):


         return visual.combination_bar_graph_av_mult(self, array_2, subarray_mult, subarray_mult_2, fig=fig)

    def multiplicity_plot(self, subarray_mult=None,fig=None):
        """
        Plot multiplicity

        Parameters
        ----------
        fig: pyplot.figure, optional
        """
            
        return visual.multiplicity_plot(self, subarray_mult, fig=fig )
    
    def multiplicity_plot_old(self, fig=None):
        """
        Plot multiplicity

        Parameters
        ----------
        fig: pyplot.figure, optional
        """
        
            
        return visual.multiplicity_plot_old(self, fig=fig)

     

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

