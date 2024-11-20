import numpy as np
import astropy.units as u

from astropy.table import Table

from . import utils as utils
from . import pointing

class Telescope:
    """
    Class containing information on a telescope
    
    Parameters
    ----------
    x: float
        x position
    y: float
        y position
    z: float
        z position
    focal: float
        focal length
    camera_radius: float
        camera radius
    """

    
    def __init__(self, _id, x, y, z, focal, camera_radius):
        
        self.id = _id
        self.x = x.to(u.m)
        self.y = y.to(u.m)
        self.z = z.to(u.m)
        self.focal = focal.to(u.m)
        self.camera_radius = camera_radius.to(u.m)
        self.alt = u.Quantity(0, u.rad)
        self.az = u.Quantity(0, u.rad)
        
        self.__make_table__()
    
    def __convert_units__(self, toDeg=True):
        """
        Convert units in the table from deg to rad, and vice versa.
    
        Parameters
        ----------
        toDeg: bool, optional
        """
        self._table = utils.deg2rad(self._table, toDeg)

    def __make_table__(self):
        """
        Make a table with input parameters

        Returns
        -------
        astropy.table
        """

        properties = [[self.id, self.x.value, self.y.value, self.z.value, 
                       self.az.value, self.alt.value,  self.zn.value, self.focal.value, 
                       self.camera_radius.value, self.fov.value, *tuple(self.pointing_vector),
                       ]]
        
        label = ["id", "x", "y", "z", "az", "alt", "zn", "focal", "radius", "fov", "p_x", "p_y", "p_z"]
        units = ["", u.m, u.m, u.m, u.rad, u.rad, u.rad, u.m, u.m, u.rad**2, "", "", ""]
        dtype = [int] + [np.float64 for i in range(len(units)-1)]
        table = Table(np.asarray(properties, dtype="object"), names=label, units=units, dtype=dtype)
        
        for col in table.columns[4:]:
            table[col].info.format = '{:.3f}'
        
        table.units = "rad"
        self._table = table

    def __update_table__(self, names, values={}):
        """
        Update a table with new parameters
    
        Parameters
        ----------
        names: array, str
            names of parameters
            If it is `pointing`, update `p_x`, `p_y`, and `p_z`.
            Otherwise update input parameters
        values: dict, optional
            values of parameters
            If `name` is in `values`, a value in a table is updated with the one
            in `values`. Otherwise, the table value matches the one in Class attributes.
            e.g.,
                if name in values.keys():
                    Telescope.table[name] = values[name]
                else:
                    Telescope.table[name] = getattr(Telescope, name)
        """
        for i, name in enumerate(names):
            if name == "pointing":
                new_val = self.pointing_vector
                for j, v in enumerate(["p_x", "p_y", "p_z"]):
                    if v in values.keys():
                        self._table[v] = values[v]
                    else:
                        self._table[v] = new_val[j]
                    self._table[v].info.format = '{:.3f}'
                
            else:
                if name in values.keys():
                    self._table[name] = values[name]
                else:
                    self._table[name] = getattr(self, name)
                self._table[name].info.format = '{:.3f}'

    def __point_to_altaz__(self, alt, az):
        """
        Make a telescope point to altitude and azumith angle
        and then update alt and az columns in a table
        
        Parameters
        ----------
        alt: float
            altitude
        az: float
            azimuth angle
        """

        self.alt = alt.to(u.rad)
        self.az = az.to(u.rad)
        if self.az < 0:
            self.az += 2*np.pi*u.rad
        self.__update_table__(["alt", "az", "zn", "pointing"])

    @property
    def table(self):
        """
        Return astropy table containing telescope information
        
        Returns
        -------
        astropy.table
        """
        return self._table

    @property
    def zn(self):
        """
        Return zenith angle in rad or deg
        
        Returns
        -------
        astropy.Quantity
        """
        if self.alt.unit == u.rad:
            return np.pi/2.*u.rad - self.alt
        else:
            return 90*u.deg - self.alt

    @property
    def fov(self):
        """
        Return a field of view (fov) in rad^2.
        
        Returns
        -------
        astropy.Quantity
        """
        return np.pi * (self.camera_radius / self.focal)**2*u.rad**2

    @property
    def position(self):
        """
        Return x, y, and z positions
        
        Returns
        -------
        array
            [x, y, z]
        """
        return np.array([self.x.to(u.m).value, self.y.to(u.m).value, self.z.to(u.m).value]*u.m)

    @property
    def pointing_vector(self):
        """
        Return pointing vectors of x, y, and z directions
        
        Returns
        -------
        array
            [p_x, p_y, p_z]
        """
        
        return pointing.alt_az_to_vector(self.alt, self.az)

