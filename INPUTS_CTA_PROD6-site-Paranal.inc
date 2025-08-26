*
* INPUTS_CTA_PROD6-site-Paranal.inc:
*    CORSIKA inputs file template component for site-specific parameters of
*    CTA-South site near Paranal. Also included: azimuth angle of primary particle.
*    since it needs to be combined with the orientation of the geomagnetic field.
*
*
* [ Site specific options for atmopheric density profile ]
* --------------------------------------------------------
*
* The selected CTA South site is now near the ESO Paranal observatory:
*   24.683429 deg South, 70.316345 deg West, altitude: 2150 m a.s.l.
#if defined(ATMOFILE)
* Custom atmospheric density profile by filename.
IACT ATMOFILE $(ATMOFILE)
ATMOSPHERE 99 Y
#elif defined(ATMOSPHERE)
* Custome atmospheric density profile by number (file atmprof<n>.dat).
ATMOSPHERE $(ATMOSPHERE) Y
#elif defined(ATMPROF_OLD)
* The atmospheric density profile as already used for prod-3 South simulations.
* Atmosphere 26 at altitude 2150.0 m:
*   Atmospheric depth at ground level: 798.805 g/cm^2
*   Density at ground level: 0.00095108 g/cm^3
*   Refractivity at ground level: 0.00021986
*   Maximum Cherenkov angle at ground level: 1.2014 deg.
* Should be slightly better for Paranal than profiles 1 (tropical) or 10 (HESS)
* but is just a model and not based on local data.
ATMOSPHERE 26 Y
#else
* Atmospheric profile for CTA South based on ECMWF weather models (original profiles by P. Munar).
# if defined(ATMPROF_SUMMER)
* An atmospheric density profile matching summer conditions.
* Fixed for self-consistency of density and thickness as well as density/pressure in extrapolation.
IACT ATMOFILE atmprof_ecmwf_south_summer_fixed.dat
# else
* An atmospheric density profile matching winter conditions.
* Fixed for self-consistency of density and thickness as well as density/pressure in extrapolation.
IACT ATMOFILE atmprof_ecmwf_south_winter_fixed.dat
# endif
* When using a custom profile by filename we need the extra atmosphere 99
* number, with or without enabling refraction (and this must come _AFTER_ IACT ATMOFILE ...!)
ATMOSPHERE 99 Y
#endif
*
* Observation level (in cm) for CTA near Paranal, adapted to lowest telescope.
OBSLEV 2147.E2
*
* [ Geomagnetic field declination and CORSIKA azimuth angles for primaries ]
* --------------------------------------------------------------------------
*
#if defined(ALIGN_B_FIELD)
* Geomagnetic field is assumed to be aligned with geographic North (array x axis).
# define AZM_FROM_SOUTH 0.
# define AZM_FROM_EAST  90.
# define AZM_FROM_NORTH 180.
# define AZM_FROM_WEST  270.
#elif defined(YEAR2020)
* Geomagnetic field aligned as originally predicted for year 2020.0.
# define AZM_FROM_SOUTH -3.433
# define AZM_FROM_EAST  86.567
# define AZM_FROM_NORTH 176.567
# define AZM_FROM_WEST  266.567
#elif defined(YEAR) && defined(HAVE_MATHEVAL)
* Correct B field declination evaluated for year $(YEAR).
# define BFDECL2020 -3.433
# define BFDECL2025 -4.533
# define BFDECL $[$(BFDECL2020)+($(BFDECL2025)-$(BFDECL2020))*($(YEAR)-2020)/(2025.-2020.)]
# define AZM_FROM_SOUTH $(BFDECL)
* From South:  PHIP = $(AZM_FROM_SOUTH)
# define AZM_FROM_EAST  $[$(BFDECL)+90]
* From East:   PHIP = $(AZM_FROM_EAST)
# define AZM_FROM_NORTH $[$(BFDECL)+180]
* From North:  PHIP = $(AZM_FROM_NORTH)
# define AZM_FROM_WEST  $[$(BFDECL)+270]
* From West:   PHIP = $(AZM_FROM_WEST)
# ifdef AZM
* AZM = $(AZM)
# endif
# ifdef PHIP
* PHIP = $(PHIP)
# endif
# if defined(AZM)
#undef PHIP
* Arrival direction of primary defined in astronomical azimuth (North->East).
#   define PHIP $[$(BFDECL)+180-$(AZM)]
* Arrival from azimuth angle $(AZM) deg. corresponds to PHIP = $(PHIP) deg.
#   undef FROM_SOUTH
#   undef FROM_EAST
#   undef FROM_NORTH
#   undef FROM_WEST
# endif
#else
* Geomagnetic field aligned as predicted for year 2025.0.
# define AZM_FROM_SOUTH -4.533
# define AZM_FROM_EAST  85.467
# define AZM_FROM_NORTH 175.467
# define AZM_FROM_WEST  265.467
#endif
* 
#ifdef FROM_SOUTH
* FROM_SOUTH = $(FROM_SOUTH)
#endif
#ifdef FROM_WEST
* FROM_WEST = $(FROM_WEST)
#endif
#ifdef FROM_NORTH
* FROM_NORTH = $(FROM_NORTH)
#endif
#ifdef FROM_EAST
* FROM_EAST = $(FROM_EAST)
#endif
*
#if defined(PHIP)
PHIP   $(PHIP) $(PHIP)       // Range of CORSIKA azimuth angles (degree): custom setting.
# ifdef AZM
* Other specific particle direction and corresponding or assigned azimuth angle of $(AZM) deg.
IACT setenv AZM $(AZM)
# else
* Other specific particle direction without known astronomical azimuth angle.
# endif
#elif defined(FROM_SOUTH)
PHIP $(AZM_FROM_SOUTH) $(AZM_FROM_SOUTH)  // CORSIKA azimuth angles (degree) for primaries coming from South.
IACT setenv AZM 180     // Corresponding astronomical azimuth, from geographical North towards East.
# define AZM 180
#elif defined(FROM_EAST)
PHIP $(AZM_FROM_EAST) $(AZM_FROM_EAST)  // CORSIKA azimuth angles (degree) for primaries coming from East.
IACT setenv AZM 90      // Corresponding astronomical azimuth, from geographical North towards East.
# define AZM 90
#elif defined(FROM_WEST)
PHIP $(AZM_FROM_WEST) $(AZM_FROM_WEST)  // CORSIKA azimuth angles (degree) for primaries coming from West.
IACT setenv AZM 270     // Corresponding astronomical azimuth, from geographical North towards East.
# define AZM 270
#elif defined(FROM_NORTH)
PHIP $(AZM_FROM_NORTH) $(AZM_FROM_NORTH)  // CORSIKA azimuth angles (degree) for primaries coming from North.
IACT setenv AZM 0       // Corresponding astronomical azimuth, from geographical North towards East.
# define AZM 0
#else
PHIP $(AZM_FROM_NORTH) $(AZM_FROM_NORTH)  // Fall-back is for primaries coming from North.
IACT setenv AZM 0
# define AZM 0
#endif
*
* Necessary rotation angle to account for B field declination (deg.)
ARRANG $(AZM_FROM_SOUTH)
*
* [ Geomagnetic field strength on H (horizontal component) / Z (-vertical) plane ]
* --------------------------------------------------------------------------------
*
* ./geomag70 IGRF12.COF 2020.00 D K2.150 -24.683429 -70.316345
*  
*  Geomag v7.0 - Jan 25, 2010 
* 
*   Model: IGRF2015 
*   Latitude: -24.68 deg
*   Longitude: -70.32 deg
*   Altitude: 2.15 km
*   Date of Interest:  2020.00
* 
*  -------------------------------------------------------------------------------
*    Date          D           I           H        X        Y        Z        F
*    (yr)      (deg min)   (deg min)     (nT)     (nT)     (nT)     (nT)     (nT)
*   2020.00    -3d  26m   -23d  33m   20924.8  20887.2  -1253.5  -9119.4  22825.7
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*    Date         dD          dI           dH       dX       dY       dZ       dF
*    (yr)      (min/yr)    (min/yr)    (nT/yr)  (nT/yr)  (nT/yr)  (nT/yr)  (nT/yr)
*   2020.00     -11.1       -9.2        -79.7    -83.7    -62.6    -32.0    -60.2
*  -------------------------------------------------------------------------------
*
* ./geomag70 IGRF13.COF 2025.00 D K2.150 -24.683429 -70.316345
*
*  Geomag v7.0 - Jan 25, 2010 
* 
*   Model: IGRF2020 
*   Latitude: -24.68 deg
*   Longitude: -70.32 deg
*   Altitude: 2.15 km
*   Date of Interest:  2025.00
* 
*  -------------------------------------------------------------------------------
*    Date          D           I           H        X        Y        Z        F
*    (yr)      (deg min)   (deg min)     (nT)     (nT)     (nT)     (nT)     (nT)
*   2025.00    -4d  32m   -24d  30m   20551.8  20487.3  -1626.7  -9367.4  22585.9
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*    Date         dD          dI           dH       dX       dY       dZ       dF
*    (yr)      (min/yr)    (min/yr)    (nT/yr)  (nT/yr)  (nT/yr)  (nT/yr)  (nT/yr)
*   2025.00     -12.3      -10.7        -75.9    -81.6    -67.2    -42.7    -51.2
*  -------------------------------------------------------------------------------
* 
* Note: geomagnetic field at height of shower maximum is slightly weaker (not accounted for).
*
#define MAGH2020 20.925
#define MAGZ2020 -9.119
#define MAGH2025 20.552
#define MAGZ2025 -9.367
#if defined(YEAR2020)
* Geomagnetic field for 2020.0:
MAGNET $(MAGH2020) $(MAGZ2020) // Magnetic field at Paranal site [H, Z] (muT), 2020.
#elif defined(YEAR) && defined(HAVE_MATHEVAL)
* Magnetic field at Paranal site [H, Z] (muT), $(YEAR)
# define MAGHYEAR $[$(MAGH2020)+($(MAGH2025)-$(MAGH2020))*($(YEAR)-2020)/(2025-2020)]
# define MAGZYEAR $[$(MAGZ2020)+($(MAGZ2025)-$(MAGZ2020))*($(YEAR)-2020)/(2025-2020)]
MAGNET $(MAGHYEAR) $(MAGZYEAR)
#else
* Geomagnetic field for 2025.0:
MAGNET $(MAGH2025) $(MAGZ2025) // Magnetic field at Paranal site [H, Z] (muT), 2025.
#endif
*
* [ End of Paranal site-specific configuration ]
*
