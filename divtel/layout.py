from .telescope import Telescope
from .array import Array
import astropy.units as u
import numpy as np

from . import utils

def LoadConfig(file, tel_id=-1, radius="degrees", frame=None, **kwargs):
    
    """
    Load the telescope configuration file
    
    Parameters
    ----------
    file: str 
        File name
    tel_id: int, optional
        If you want to load only a single telescope,
        you can set this parameter (defalut: -1)
    radius: str, optional
        Define the unit of camera radius
        either `meters` or `degrees` (default: degrees).
    frame: class.CTA_Info, optional
    kwargs: args for class.Array
        
    Returns
    -------
    class.Array
    """

    with open(file, "r") as f:
        
        tels = []
        for i, line in enumerate(f.readlines()):
            line = np.asarray(line.split()).astype("float")

            if (radius!="meters"):
                line[4] = utils.convert_radius(line[4]*u.deg, line[3], toDeg=False)
            
            coord = [x*u.m for x in line]

            tel = Telescope(i+1, coord[0],coord[1],coord[2],coord[3],coord[4])
            tels.append(tel)

    if tel_id == -1:
        return Array(tels, frame=frame, **kwargs)
    else:
        return tels[tel_id-1]
 
def LoadTable(table, tel_id=-1, radius="degrees", frame=None, **kwargs):
    
    """
    Load the telescope configuration from table
    This function is meant to create subarrays starting from the general array
    
    Parameters
    ----------
    table: astropy table
        File name
    tel_id: int, optional
        If you want to load only a single telescope,
        you can set this parameter (defalut: -1)
    radius: str, optional
        Define the unit of camera radius
        either `meters` or `degrees` (default: degrees).
    frame: class.CTA_Info, optional
    kwargs: args for class.Array
        
    Returns
    -------
    class.Array
    """

    tels = []
    for i in range(len(table)):
            #line = np.asarray(line.split()).astype("float")

            if (table['radius'].unit!=u.m):
                table['radius'][i] = utils.convert_radius(table['radius'][i]*u.deg, table['focal'][i], toDeg=False)
            
            

            tel = Telescope(table['id'][i], table['x'][i]*u.m,table['y'][i]*u.m,table['z'][i]*u.m,table['focal'][i]*u.m,table['radius'][i]*u.m)
            tels.append(tel)

    if tel_id == -1:
        return Array(tels, frame=frame, **kwargs)
    else:
        return tels[tel_id-1]
