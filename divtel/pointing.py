"""
Functions to define telescopes pointings
We use the same reference frame as simtel_array:
X is pointing North
Y is pointing East
Z is pointing upward
Az is taken clock-wise from X (towards Y) and between 0 and 180 degrees
Alt is taken from ground (towards Z) and between 0 and 90 degrees
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS


def alt_az_to_vector(alt, az):
    """
    Compute a pointing vector from an alt, az pointing direction
    
    Parameters
    ----------
    alt: float
        angle in rad
    az: float
        angle in rad
    Returns
    -------
    array
        [x, y, z]
    """

    x = np.cos(alt.to(u.rad)) * np.cos(az.to(u.rad))
    y = -np.cos(alt.to(u.rad)) * np.sin(az.to(u.rad))
    z = np.sin(alt.to(u.rad))
    return np.array([x, y, z])

def _norm_div(div, scale=100):
    """
    Transformation function from div parameter to norm 
    to compute the position of G 

    Parameters
    ----------
    div: float
    scale: float
        telescope distance from barycenter at which 
        div = divergence_angle/90deg
    
    Returns
    -------
    float
    """
    return scale/np.tan(np.arcsin(div))

def pointG_position(barycenter, div, alt_mean, az_mean):
    """
    Compute the position of G for the pointing
    
    Parameters
    ----------
    barycenter: np.array([x,y,z])
        position of the barycenter of the array
    div: float
    alt_mean: astropy.Quantity
        mean pointing altitude in radians from which to diverge
    az_mean: astropy.Quantity
        mean pointing azimuth in radians from which to diverge
    
    Returns
    -------
    array 
        [Gx, Gy, Gz]
    """
    norm = _norm_div(div)
    Gx = barycenter[0] - norm * np.cos(alt_mean) * np.cos(az_mean)
    Gy = barycenter[1] + norm * np.cos(alt_mean) * np.sin(az_mean)
    Gz = barycenter[2] - norm * np.sin(alt_mean)
    return np.array([Gx, Gy, Gz])

def tel_div_pointing(tel_position, G):
    """
    Divergent pointing to a point G.

    Parameters
    ----------
    tel_position: array
        telescope coordinates, [x, y, z]
    G: array
        [Gx, Gy, Gz]

    Returns
    -------
    alt: float
        altitude
    az: float
        azumith angle
    """
    GT = np.sqrt(((tel_position - G) ** 2).sum())
    alt = np.abs(np.arcsin((tel_position[2] - G[2]) / GT))
    az = np.arctan2(-(tel_position[1] - G[1]), (tel_position[0] - G[0]))
    return alt, az


def pointing_coord(table, frame, icrs=False):
    """
    Sky coordinate of each telescope with a given frame.

    Parameters
    ----------
    table: astropy.table
        table containing `alt` and `az` of each telescope
    frame: class.CTA_Info
        frame containing information of site and obsrvation time
    icrs: bool, optional
        If True, it returns ICRS coordinate

    Returns
    -------
    astropy.coordinates.SkyCoord
    """
    coord = SkyCoord(alt=table["alt"], az=table["az"], frame=frame.altaz, unit=table["alt"].unit)
    
    if icrs:
        return coord.transform_to(ICRS())
    else:
        return coord

