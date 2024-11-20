import astropy.units as u
import numpy as np
import sys
from astropy.coordinates import SkyCoord, get_icrs_coordinates
from divtel import *
import healpy as hp
import tqdm
import yaml
import argparse
# -----------------------------------------------
parser = argparse.ArgumentParser(description='Checking FoV evolution for different RaDec positions for divergent pointing')
parser.add_argument('-f', '--config', required=True, type=str, help='configuration yaml file')

#--------------------------------------------------READING configuration file
cf = parser.parse_args().config

with open(cf) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

daytime=cfg['system']['daytime']
site=cfg['system']['site']
configuration=cfg['system']['configuration']
obs_h=cfg['obs_hours']
# ------------------------------- load config_file
if site == "north":
    config_file = "/home/irene/ctasoft/divtel_irene/config/LaPalmaArrayPositions_divProd6.txt" 
elif site =="south":
    config_file = "/home/irene/ctasoft/divtel_irene/config/Paranal_prod6_alpha4LSTs_reduced_list.txt"
    

# ------------------------------------------------------
m_cut=cfg['system']['multiplicity_cut']
outfile=cfg['outfile']
# ------------------List of default stars
if cfg['star']==None:
    if site == 'north':
        stars=['mirfak','errai','sirius', 'alpha cmi','pollux','epsilon umi',
               'kochab','elnath', 'betelgeuse','capella','bellatrix','rigel',
               'aldebaran','menkar','regulus','alpha cas','mirach','hamal',
               'gamma eri','alpha and','beta per','electra','omicron uma',
               'epsilon leo','sao 6487','sao 168460','polaris','sao 136871',
               'dubhe','merak','52 uma','54 uma','rho pup','delta umi',
               '35 hya','gamma uma','alula borealis', '30 boo',
               'theta leo','gamma boo', 'mizar', 'alpha com','70 vir', '79 vir',
               'vega','alpha oph', 'marfik', 'eta oph', 'antares','79 vir',
               'delta her', 'gamma dra','eta dra','alpha dra', 'alpha crb',
               'alpha ser','delta scorpio','deneb','17 aql']

    elif site =='south':
        stars=['canopus', 'alpha eri', 'alpha psa', 'acrux', 'beta car',
               'aspidiske', 'sirius', 'procyon', 'betelgeuse','bellatrix',
               'rigel','aldebaran', 'capella','beta cet', 'alpha phe',
               'theta1 eri','hamal','pleiades','beta phe', 'beta lep',
               'delta cma', 'spica','gamma1 leo', 'gamma crv', 'beta tau',
               'beta leo','alpha lyn', 'lmc','adhara']

else:
    stars=[cfg['star']]
# ------------------------------------------------------------------------------
if cfg['system']['divergence']==None:
    divergence=[0.0022,0.0043,0.008,0.01135,0.01453]
else:
    divergence=[cfg['system']['divergence']]
#-------------------------------------------------------------------------------


results = {}
star_coord=[]

print('\nLOADING STAR POSITIONS FROM CATALOG')
print('\n')
for name in stars:
    star_coord.append(get_icrs_coordinates(name))
print('Starting simulation')
for i,star in enumerate(star_coord):
    name=stars[i]
    results={}
    cta = CTA_Info(site,daytime)
    print (f'\nstar:{name}')
    results[name]={}
    star = get_icrs_coordinates(name)
    star_altaz=star.transform_to(cta.altaz)

    cta.set_source_loc(ra=star.ra, dec=star.dec)
    initial_time=cta.t_obs
    while cta.source.alt<=24*u.deg and cta.t_obs<=initial_time+1*u.day:

        cta.update(delta_t = 30*u.min)
        cta.set_source_loc(ra=star.ra, dec=star.dec)

    if cta.t_obs>=initial_time+1*u.day:
        print(f'your source is never visible from {cta.site}')
        continue
    print(f'start time: {cta.t_obs}')
    #print ("source:", cta.source)v_div{div}
    results[name]['ra_dec']=[star.ra.deg,star.dec.deg]
    for div in divergence:
        print (f'\n\tdivergence:{div}')
        results[name][div]={}
        results[name][div]['alt']=[]
        results[name][div]['az']=[]
        results[name][div]['obsname']=[]
        results[name][div]['obstime']=[]
        results[name][div]['hFoV_track']=[]
        results[name][div]['m_ave_track']=[]
        results[name][div]['hFoV_div']=[]
        results[name][div]['m_ave_div']=[]
        
        #pointing to source
        array =  LoadConfig(config_file, frame=cta, pointing2src=True)
        #apply divergence
        array.divergent_pointing(div)
        #array.hFoV(m_cut=multiplicity)
        initial_pointing_dir=array.get_pointing_coord(icrs=True)
        nside = 512
        map_multiplicity = np.zeros(hp.nside2npix(nside), dtype=np.int8)
        counter = np.arange(0, hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
        coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        array.table.units="deg"
        for i in tqdm.tqdm(range(len(array.telescopes))):
            pointing = SkyCoord(ra=array.table['az'][i], dec=array.table['alt'][i], unit='deg')
            r_fov = np.arctan((array.telescopes[i].camera_radius/array.telescopes[i].focal).to(u.dimensionless_unscaled)).to(u.deg)
            mask = coordinate.separation(pointing) < r_fov
            map_multiplicity[mask] += 1

        mask_fov = map_multiplicity>0
        mask_fov_eff = map_multiplicity>3
        if cfg['verbose']==True:
            print('hfov:', hp.nside2pixarea(nside, True)*np.sum(mask_fov),
                  'hfov_eff:', hp.nside2pixarea(nside, True)*np.sum(mask_fov_eff),
                  np.sum(mask_fov_eff)/np.sum(mask_fov))
            print('m_ave:',np.mean(map_multiplicity[mask_fov]),
                  'm_ave(cut3):',np.mean(map_multiplicity[mask_fov_eff]))


        results[name][div]['alt'].append(cta.source.alt.deg)
        results[name][div]['az'].append(cta.source.az.deg)
        results[name][div]['obsname'].append(cta.observer.name)
        results[name][div]['obstime'].append(array.frame.t_obs.value)
        results[name][div]['hFoV_track'].append(hp.nside2pixarea(nside, True)*np.sum(mask_fov))
        results[name][div]['m_ave_track'].append(np.mean(map_multiplicity[mask_fov]))
        results[name][div]['hFoV_div'].append(hp.nside2pixarea(nside, True)*np.sum(mask_fov))
        results[name][div]['m_ave_div'].append(np.mean(map_multiplicity[mask_fov]))

        

        for dt in range(int(obs_h*3)): #
            print('\n')
            initial_pointing_dir=array.get_pointing_coord(icrs=True)

            array.update_frame(delta_t = 20*u.min, verbose=True)
            new_frame=array.frame.altaz
            star_altaz=star.transform_to(new_frame)
            #print(array.table[0])

            if star_altaz.alt.deg >=24:

                final_pointing=initial_pointing_dir.transform_to(new_frame)
                nside = 512
                map_multiplicity = np.zeros(hp.nside2npix(nside), dtype=np.int8)
                map_multiplicity_div = np.zeros(hp.nside2npix(nside), dtype=np.int8)
                counter = np.arange(0, hp.nside2npix(nside))
                ra, dec = hp.pix2ang(nside, counter, True, lonlat=True)
                coordinate_div = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                coordinate = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                array.table.units="deg"
                for i in tqdm.tqdm(range(len(array.telescopes))):

                    #repointing_telescopes
                    pointing_div = SkyCoord(ra=array.table['az'][i], dec=array.table['alt'][i], unit='deg')
                    r_fov_div = np.arctan((array.telescopes[i].camera_radius/array.telescopes[i].focal).to(u.dimensionless_unscaled)).to(u.deg)
                    mask_div = coordinate_div.separation(pointing_div) < r_fov_div
                    map_multiplicity_div[mask_div] += 1

                    #tracking_initial_position
                    pointing = SkyCoord(ra=final_pointing[i].az, dec=final_pointing[i].alt, unit='deg')
                    r_fov = np.arctan((array.telescopes[i].camera_radius/array.telescopes[i].focal).to(u.dimensionless_unscaled)).to(u.deg)
                    mask = coordinate.separation(pointing) < r_fov
                    map_multiplicity[mask] += 1


                mask_fov = map_multiplicity>0
                mask_fov_eff = map_multiplicity>3
                if cfg['verbose']==True:
                    print('hfov_track:', hp.nside2pixarea(nside, True)*np.sum(mask_fov),
                          'hfov_eff_track:', hp.nside2pixarea(nside, True)*np.sum(mask_fov_eff),
                          np.sum(mask_fov_eff)/np.sum(mask_fov))
                    print('m_ave_track:',np.mean(map_multiplicity[mask_fov]),
                          'm_ave(cut3)_track:',np.mean(map_multiplicity[mask_fov_eff]))
               
                mask_fov_div = map_multiplicity_div>0
                mask_fov_eff_div = map_multiplicity_div>3
                if cfg['verbose']==True:
                    print('hfov_div:', hp.nside2pixarea(nside, True)*np.sum(mask_fov_div),
                          'hfov_eff_div:', hp.nside2pixarea(nside, True)*np.sum(mask_fov_eff_div),
                          np.sum(mask_fov_eff_div)/np.sum(mask_fov_div))
                    print('m_ave:',np.mean(map_multiplicity_div[mask_fov_div]),
                          'm_ave(cut3):',np.mean(map_multiplicity_div[mask_fov_eff_div]))

                results[name][div]['alt'].append(array.pointing['alt'].value)
                results[name][div]['az'].append(array.pointing['az'].value)
                results[name][div]['obsname'].append(cta.observer.name)
                results[name][div]['obstime'].append(array.frame.t_obs.value)
                results[name][div]['hFoV_track'].append( hp.nside2pixarea(nside, True)*np.sum(mask_fov))
                results[name][div]['m_ave_track'].append(np.mean(map_multiplicity[mask_fov]))
                results[name][div]['hFoV_div'].append(hp.nside2pixarea(nside, True)*np.sum(mask_fov_div))
                results[name][div]['m_ave_div'].append(np.mean(map_multiplicity_div[mask_fov_div]))


if len(stars)==1:
    np.save(f'plots/{outfile}_{stars[0]}_{daytime}.npy' , results)
else:
    np.save(f'plots/{outfile}_{daytime}.npy' , results)
