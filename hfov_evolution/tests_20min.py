from ctadiv import *
import matplotlib.pyplot as plt
from astropy.coordinates import get_icrs_coordinates
import yaml
import argparse
import matplotlib as mpl
from descartes import PolygonPatch
from shapely.ops import unary_union, polygonize
from shapely.geometry import mapping, Polygon, Point, LineString
#---------------------------------------------------

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

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
if site == 'north':
    if configuration == 'baseline':
        config_file="./config/layout-3AL4M15-5.txt"
    elif configuration == 'alpha':
        config_file="./config/layout_alpha_north.txt"
elif site == 'south':
    if configuration == 'baseline':
        config_file="./config/layout_paranal_HB9.txt"
    elif configuration == 'alpha':
        config_file="./config/layout_paranal_alpha.txt"
# ------------------------------------------------------
m_cut=cfg['system']['multiplicity_cut']
outfile=cfg['outfile']
# ------------------List of default stars
if cfg['star']==None:
    stars=['mirfak','errai','sirius']#, 'alpha cmi','pollux','epsilon umi',
           #'kochab','elnath', 'betelgeuse','capella','bellatrix','rigel',
           #'aldebaran','menkar','regulus','alpha cas','mirach','hamal',
           #'gamma eri','alpha and','beta per','electra','omicron uma',
           #'epsilon leo','sao 6487','sao 168460','polaris','sao 136871',
           #'dubhe','merak','52 uma','54 uma','rho pup','delta umi',
           #'alpha cet','35 hya','gamma uma','alula borealis', '30 boo',
           #'theta leo','gamma boo', 'mizar', 'alpha com','70 vir', '79 vir',
           #'vega','alpha oph', 'marfik', 'eta oph', 'antares','79 vir',
           #'delta her', 'gamma dra','eta dra','alpha dra', 'alpha crb',
           #'alpha ser','delta scorpio','deneb','17 aql']


else:
    stars=[cfg['star']]
# ------------------------------------------------------------------------------
if cfg['system']['divergence']==None:
    divergence=[0.0022,0.0043,0.008,0.01135,0.01453]
else:
    divergence=[cfg['system']['divergence']]
#-------------------------------------------------------------------------------
if cfg['header']==True:
    details=('name,ra(deg),dec(deg),alt(deg),az(deg),daytime,site,div,fov(deg2),average multiplicity,tracking')
    append_new_line(f'plots/{outfile}.txt', details)



results = {}
star_coord=[]

print('\nLOADING STAR POSITIONS FROM CATALOG')
print('\n')
for name in stars:
    star_coord.append(get_icrs_coordinates(name))
print('Starting simulation')
for i,star in enumerate(star_coord):
    name=stars[i]
    cta = CTA_Info(site,daytime)
    print (f'\nstar:{name}')
    results[name]={}
    #star = get_icrs_coordinates(name)
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
        results[name][div]['hfov_diff']=[]
        results[name][div]['m_ave_diff']=[]

        #pointing to source
        array =  LoadConfig(config_file, frame=cta, pointing2src=True)
        #apply divergence
        array.divergent_pointing(div)
        #array.hFoV(m_cut=multiplicity)
        initial_pointing_dir=array.get_pointing_coord(icrs=True)
        if cfg['verbose']==True:
            print(f'\n\thFoV:{array.hFoV(m_cut=m_cut)}, average multiplicity:{array.hFoV(m_cut=m_cut,return_multiplicity=True)[1]}')


        results[name][div]['alt'].append(cta.source.alt.deg)
        results[name][div]['az'].append(cta.source.az.deg)
        results[name][div]['obsname'].append(cta.observer.name)
        results[name][div]['obstime'].append(array.frame.t_obs.value)
        results[name][div]['hFoV_track'].append(array.hFoV(m_cut=m_cut).value)
        results[name][div]['m_ave_track'].append(array.hFoV(m_cut=m_cut,return_multiplicity=True)[1])
        results[name][div]['hFoV_div'].append(array.hFoV(m_cut=m_cut).value)
        results[name][div]['m_ave_div'].append(array.hFoV(m_cut=m_cut,return_multiplicity=True)[1])
        results[name][div]['hfov_diff'].append(0)
        results[name][div]['m_ave_diff'].append(0)
        #results[name][div]['pointing_diff'].append(0)

        details=(f'{name},{star.ra.deg},{star.dec.deg},{star_altaz.alt.deg},{star_altaz.az.deg},{array.frame.t_obs.value},{cta.observer.name},{div},{array.hFoV().value},{array.hFoV(return_multiplicity=True)[1]},initial_point')
        append_new_line(f'plots/{outfile}.txt', details)

        for dt in range(int(obs_h*3)): #
            print('\n')

            array.update_frame(delta_t = 20*u.min, verbose=True)

            new_frame=array.frame.altaz

            star_altaz=star.transform_to(new_frame)
            if star_altaz.alt.deg >=24:
                if cfg['verbose']==True:
                    print(f'\n\thFoV:{array.hFoV()}, average multiplicity:{array.hFoV(return_multiplicity=True)[1]}')


                details=(f'{name},{star.ra.deg},{star.dec.deg},{star_altaz.alt.deg},{star_altaz.az.deg},{array.frame.t_obs.value},{cta.observer.name},{div},{array.hFoV().value},{array.hFoV(return_multiplicity=True)[1]},False')
                append_new_line(f'plots/{outfile}.txt', details)

                final_pointing=initial_pointing_dir.transform_to(new_frame)
                # --------- messy part - tracking the stars
                polygons = {}
                for i,pointing in enumerate(final_pointing):
                    tels_points = pointing
                    if max(final_pointing[:].az.deg)-min(final_pointing[:].az.deg) > 180:


                        if tels_points.az.degree < 180:
                            polygons[i]=(Point(tels_points.az.degree, tels_points.alt.degree).buffer(array.table["radius"][i]))
                        else:
                            polygons[i]=(Point(tels_points.az.degree-360, tels_points.alt.degree).buffer(array.table["radius"][i]))

                    else:

                        polygons[i] = Point(tels_points.az.degree, tels_points.alt.degree).buffer(array.table['radius'][i])

                rings = [LineString(list(pol.exterior.coords)) for pol in polygons.values()]
                union = unary_union(rings)
                result = {counter:geom for counter, geom in enumerate(polygonize(union))}

                ori = list(polygons.values())
                res = list(result.values())

                dict_count_overlaps = {}
                for i in range(len(res)):
                    dict_count_overlaps[i] = 0
                    for j in range(len(ori)):
                         if np.isclose(res[i].difference(ori[j]).area, 0):
                            dict_count_overlaps[i] +=1
                             #print(f"res_{colors[i]}, orig_{j+1}")


                max_multiplicity = max(dict_count_overlaps.values())
                overlaps_nocut = np.array(list(dict_count_overlaps.values()))
                #print(len(res), len(overlaps_nocut))
                hfov = []
                overlaps=[]
                for i,patchsky in enumerate(res):

                    if overlaps_nocut[i]>m_cut-1:
                        overlaps.append(overlaps_nocut[i])
                        hfov.append(patchsky.area)

                hfov = np.array(hfov)
                overlaps = np.array(overlaps)
                average_overlap = np.average(overlaps, weights=hfov)
                variance = np.average((overlaps-average_overlap)**2, weights=hfov)

                results[name][div]['alt'].append(array.pointing['alt'].value)
                results[name][div]['az'].append(array.pointing['az'].value)
                results[name][div]['obsname'].append(cta.observer.name)
                results[name][div]['obstime'].append(array.frame.t_obs.value)
                results[name][div]['hFoV_track'].append(hfov.sum())
                results[name][div]['m_ave_track'].append(average_overlap)
                results[name][div]['hFoV_div'].append(array.hFoV(m_cut=m_cut).value)
                results[name][div]['m_ave_div'].append(array.hFoV(m_cut=m_cut,return_multiplicity=True)[1])
                results[name][div]['hfov_diff'].append(array.hFoV(m_cut=m_cut).value-hfov.sum())
                results[name][div]['m_ave_diff'].append(array.hFoV(m_cut=m_cut,return_multiplicity=True)[1]-average_overlap)
                #results[name][div]['pointing_diff'].append(
                details=(f'{name},{star.ra.deg},{star.dec.deg},{star_altaz.alt.deg},{star_altaz.az.deg},{cta.t_obs.value},{cta.observer.name},{div},{array.hFoV().value},{hfov.sum()},{average_overlap}')
                append_new_line(f'plots/{outfile}.txt', details)

                initial_pointing_dir=array.get_pointing_coord(icrs=True)
                if cfg['verbose']==True:
                    print(f'\thFoV:{hfov.sum()}, average multiplicity:{average_overlap}')




if len(stars)==1:
    np.save(f'plots/{outfile}_{stars[0]}_{daytime}.npy' , results)
else:
    np.save(f'plots/{outfile}_{daytime}.npy' , results)
