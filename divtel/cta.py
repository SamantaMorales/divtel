import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.time import Time
from astroplan import Observer

import astropy.units as u

from astropy.coordinates import get_moon
from astropy.coordinates import get_sun

from astropy.visualization import astropy_mpl_style, quantity_support

plt.style.use(astropy_mpl_style)
quantity_support()

    
class CTA_Info:

    """
    CTA frame based on its location and a given observation time.
    
    Parameters
    ----------
    site: str, optional 
        CTA location; North (default), South, Roque de los Muchachos, or Paranal.
    time: str, optional
        Observation time (yyyy-MM-ddThh:mm:ss). Default is astropy.time.Time.now()
    verbose: bool, optional

    Returns
    -------
    class
    """

    def __init__(self, site="North", time = None, verbose=True):
        
        if time is None:
            time = Time.now()

        self._site = site
        self.site_def(site)
        self._observer = Observer(location=self.loc, name=self.name)
        self._t_obs = Time(time, format='isot', scale='utc')
        self._source = None
        self._timestep = np.linspace(-12, 12, 100)*u.hour
        if verbose:
            self.info

    @property
    def info(self):
        """
        Print information
        """
        print("Observer         : ", self.observer.name)
        print("Location         : ", self.site,  ",", self.loc.to(u.km))
        print("Observation time : ", self.t_obs)

    @property
    def loc(self):
        """
        Coordinates of observatory

        Returns
        -------
        astropy.coordinates.EarthLocation.from_geodetic
        """

        if self.site.lower() in ('north', 'roque de los muchachos'):
            site_coords = EarthLocation.from_geodetic('342.1184', '28.7606', 2326. * u.meter)
        elif self.site.lower() in ('south', 'paranal'):
            site_coords = EarthLocation.from_geodetic('289.5972', '-24.6253', 2635. * u.meter)
        else:
            raise Warning(f"{site} is not a valid site choice")
        return site_coords
    
    @property
    def t_obs(self):
        """
        Observation time

        Returns
        -------
        astropy.time
        """
        return self._t_obs

    @property
    def observer(self):
        """
        A container class for information about an observer’s location and environment.
        
        Returns
        -------
        astroplan.Observer
        """
        return self._observer
    
    @property
    def altaz(self):
        """
        An AltAz frame based on the observation time and location.
        
        Returns
        -------
        astropy.astropy.coordinates.AltAz
        """
        return AltAz(obstime=self.t_obs, location=self.loc)

    @property
    def site(self):
        """
        The location of observatory; Roque de los Muchachos or Paranal
        
        Returns
        -------
        str
        """
        return self._site

    @property
    def name(self):
        """
        The name of observatory; CTA North or CTA South

        Returns
        -------
        str
        """
        return self._name

    @property
    def source(self):
        """
        Return the sky coordinate of a source, which is defined by
        CTA_Info.set_source_loc.

        Returns
        -------
        astropy.coordinates.SkyCoord
        """
        return self._source

    def _time_bin(self, timestep):
        """
        Return an array of time, t_obs+timestep.
        
        Parameters
        ----------
        timestep: array, astropy.Quantity
            If timestep does not have an unit, it is assumed to be hour.
            default: np.linspace(-12, 12, 100)*u.hour
        """
        if timestep is not None:
            if type(timestep) != u.Quantity:
                print("[Warning] The unit of timestep is assumed to be 'hour'.")
                self._timestep = timestep*u.hour
            else:
                self._timestep = timestep
        
        time = self.t_obs+self._timestep
        return time

    def site_def(self, site):
        """
        Define a site of observatory.
        
        Parameters
        ----------
        site: str 
              CTA location; North (default), South, Roque de los Muchachos, or Paranal.
        """
        if self.site.lower() in ('north', 'roque de los muchachos'):
            self._site = "Roque de los Muchachos"
            self._name = "CTA North"
        elif self.site.lower() in ('south', 'paranal'):
            self._site = "Paranal"
            self._name = "CTA South"
        else:
            raise Warning(f"{site} is not a valid site choice")

    def get_sun_loc(self, timespan=False, timestep=None):
        """
        Return Sun location given time and location.
        
        Parameters
        ----------
        timespan: bool, optional 
            If True, it returns the location of Moon as a function of time
        timestep: array, astropy.Quantity, optional 
            When timespan is True, timestep can be optioanlly used.
            See CTA_Info._time_bin

        Returns
        -------
        astropy.coordinates.get_sun
        """

        if timespan:
            time = self._time_bin(timestep)
        else:
            time = self.t_obs

        frame = AltAz(obstime=time, location=self.loc)
        return get_sun(time).transform_to(frame)

    def get_moon_loc(self, timespan=False, timestep=None):
        """
        Return Moon location given time and location.
        
        Parameters
        ----------
        timespan: bool, optional 
            If True, it returns the location of Moon as a function of time
        timestep: array, astropy.Quantity, optional 
            When timespan is True, timestep can be optioanlly used.
            See CTA_Info._time_bin

        Returns
        -------
        astropy.coordinates.get_moon
        """
        if timespan:
            time = self._time_bin(timestep)
        else:
            time = self.t_obs

        frame = AltAz(obstime=time, location=self.loc)
        return get_moon(time).transform_to(frame)

    def set_source_loc(self, ra, dec, timespan=False, timestep=None, units='deg'):
        """
        Set a source location with a given frame.
        
        Parameters
        ----------
        ra: float
            Right ascension of a source
        dec: float 
            Declination of a source
        units: str
            Units of RA and DEC; either deg (default) or rad
        timespan: bool, optional 
            If True, it returns the location of Moon as a function of time
        timestep: array, astropy.Quantity, optional 
            When timespan is True, timestep can be optioanlly used.
            See CTA_Info._time_bin
        Returns
        -------
        astropy.coordinates.SkyCoord
        """

        source_radec = SkyCoord(ra=ra, dec=dec, frame=ICRS, unit=units)
        
        if timespan:
            time = self._time_bin(timestep)
            frame = AltAz(obstime=time, location=self.loc)
            src = source_radec.transform_to(frame)
        else:
            src = source_radec.transform_to(self.altaz)
            self._source = src
        
        return src
    
    def update(self, site=None, time=None, delta_t=None, verbose=False):
        """
        Update site and/or observation time.
        
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
        if site is not None:
            self._site = site
            self.site_def(site)
            self._observer = Observer(location=self.loc, name=self.name)

        elif time is not None:
            self._t_obs = Time(time, format='isot', scale='utc')

        elif delta_t is not None:
            if type(delta_t) != u.Quantity:
                print("[Warning] The units of delta_t is assumed to be 'hour'.")
                delta_t = delta_t*u.hour
            
            self._t_obs = Time(self.t_obs+delta_t, format='isot', scale='utc')            
        
        if verbose:
            self.info

    def navigation_plot(self, timestep = None, **kwargs):
        """
        Navigation plot, which shows azimuth angles of Sun, Moon, and a source as a function of time.
        
        Parameters
        ----------
        timestep: array, astropy.Quantity, optional 
            When timespan is True, timestep can be optioanlly used.
            See CTA_Info._time_bin
        kwargs: dict, optional
             args for either `CTA_Info.set_source_loc` or `pyplot.scatter`.
        """
        if timestep is not None:
            if type(timestep) != u.Quantity:
                print("[Warning] The unit of timestep is assumed to be 'hour'.")
                self._timestep = timestep*u.hour
            else:
                self._timestep = timestep

        sun = self.get_sun_loc(timespan=True)
        moon = self.get_moon_loc(timespan=True)
        plt.plot(self._timestep, sun.alt, color='r', label='Sun')
        plt.plot(self._timestep, moon.alt, color=[0.75]*3, ls='--', label='Moon')

        ra = kwargs.pop("ra", None)
        dec = kwargs.pop("dec", None)
        units = kwargs.pop("units", "deg")
        if (ra is not None) and (dec is not None):
            src = self.set_source_loc(ra=ra, dec=dec, timespan=True, units=units)
        else:
            src = self.set_source_loc(ra=self.source.icrs.ra, dec=self.source.icrs.dec, 
                                      timespan=True, units=units)

        plt.plot(self._timestep, src.alt, lw=0.5, alpha=0.5, color="orange")
        plt.scatter(self._timestep, src.alt,
                    c= src.az.value, s=8,
                    cmap='viridis',**kwargs)

        plt.fill_between(self._timestep, 0, 90*u.deg,
                         sun.alt < -0*u.deg, color='0.5', zorder=0)
        plt.fill_between(self._timestep, 0*u.deg, 90*u.deg,
                         sun.alt < -18*u.deg, color='k', zorder=0)

        plt.colorbar().set_label('Azimuth [deg]')
        plt.legend(loc='upper left')
        plt.xlim(-12*u.hour, 12*u.hour)
        plt.xticks((np.arange(13)*2-12)*u.hour)
        plt.ylim(0*u.deg, 90*u.deg)
        plt.xlabel('Hours from EDT Midnight')
        plt.ylabel('Altitude [deg]')
        plt.show(block=False)
        return plt
