import numpy as np
import matplotlib.pyplot as plt

from shapely.ops import unary_union, polygonize
from shapely.geometry import LineString, Point

import astropy.units as u

def calc_mean(table, columns):
    """
    Calculate a mean value of columns

    Parameters
    ----------
    table: astropy.table
    columns: str or array
        names of columns

    Returns
    -------
    float
    """

    if np.size(columns) == 1:
        columns = [columns]
    
    mean = []
    for column in columns:
        if column in ["p_x", "p_y", "p_z"]:
            mean.append(np.average(table[column], weights=table["d_tel"]))
        else:
            mean.append(np.mean(table[column]))
    return tuple(mean)
    
def deg2rad(table, toDeg=False):
    """
    Convert units in a table from rad to deg and vice versa.
    `az`, `alt`, `zn`, `radius`, `fov` units are changed.

    Parameters
    ----------
    table: astropy.table
    toDeg: bool, optional
        If True, the output table is in `deg`. Otherwise, in `rad`.

    Returns
    -------
    astropy.table
    """
    if toDeg:
        for par in ["az", "alt", "zn"]:
            table[par] = table[par].to(u.deg)
            table[par].info.format = '{:.3f}'

        table["radius"] = convert_radius(table["radius"], table["focal"], toDeg=toDeg)
        table["radius"].info.format = '{:.3f}'

        table["fov"]    = table["fov"].to(u.deg**2)
        table["fov"].info.format = '{:.3f}'
    else:
        for par in ["az", "alt", "zn"]:
            table[par] = table[par].to(u.rad)
            table[par].info.format = '{:.3f}'
        
        table["radius"] = convert_radius(table["radius"], table["focal"], toDeg=toDeg)
        table["radius"].info.format = '{:.3f}'
        
        table["fov"]    = table["fov"].to(u.rad**2)
        table["fov"].info.format = '{:.3f}'
    return table

def convert_radius(radius, focal, toDeg=False):
    """
    Convert the unit of camera radius from degree to meter, and vice versa.

    Parameters
    ----------
    radius: float, astropy.Quantity
        camera radius of a telecope
    focal: float, astropy.Quantity 
        focal length of a telescope
    toDeg: bool, optional
        If True, the output is in degree.
        If False, the output is in meter.

    Returns
    -------
    astropy.Quantity
    """
    if toDeg and radius.unit == u.m:
        temp = np.arctan(np.asarray(radius/focal))
        temp = temp*u.rad
        radius = temp.to(u.deg)
    elif not(toDeg) and radius.unit == u.deg:        
        temp = radius.to(u.rad)
        radius= np.tan(temp.value)*focal
    return radius

def hfov_from_table(table, m_cut=0, return_multiplicity=False, full_output=False):

        if table['alt'].unit == 'rad':
            table['alt']=table['alt']*180/np.pi
            table['az']=table['az']*180/np.pi
      
       
            
        if max(table["az"])-min(table["az"]) > 180:
            polygons = []
            
            for az, alt, r in zip(table['az'], table['alt'], table["radius"]):
                if az < 180:
                    polygons.append(Point(az, alt).buffer(r))
                else:
                    polygons.append(Point(az-360, alt).buffer(r))
        else:
            polygons = [Point(az, alt).buffer(r) for az, alt, r in zip(table['az'], table['alt'], table["radius"])]

        union = unary_union([LineString(list(pol.exterior.coords)) for pol in polygons])
        geoms = [geom for geom in polygonize(union)]
        hfov = [geom.area for geom in geoms]

        count_overlaps = np.zeros(len(geoms))
        for i, geom in enumerate(geoms):
            count_overlaps[i] = sum([1 for pol in polygons if abs(geom.difference(pol).area)<1e-5])

        hfov = np.array(hfov)

        # multiplicity associated with each patch
        overlaps = np.array(count_overlaps)
        multiplicity = np.array([[i, hfov[overlaps==i].sum()] for i in set(overlaps)])
        eff_overlaps=[]
        eff_geoms=[]
        for i in range(len(overlaps)):
            if overlaps[i]>m_cut:
                eff_overlaps.append(overlaps[i])
                eff_geoms.append(geoms[i])

        fov = sum(multiplicity[:,1][multiplicity[:,0]>=m_cut])*u.deg**2

        if full_output:
            return multiplicity, eff_overlaps, eff_geoms
        elif return_multiplicity:
            m_ave = np.average(multiplicity[:,0], weights=multiplicity[:,1])
            m_var = np.average((multiplicity[:,0]-m_ave)**2, weights=multiplicity[:,1])
            return fov, m_ave, m_var
        else:
            return fov

