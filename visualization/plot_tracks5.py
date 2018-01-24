from teca import *
import numpy as np
import sys, os

# ------------------------------------------------------------
# part 1
# ------------------------------------------------------------
# the first part of the demo shows how to load data output by
# TECA. the second part shows an example of analyzing the data
# using matplotlib.
# ------------------------------------------------------------

# data can be passed on command line, if not assume default
# demo data
demo_dir = os.path.dirname(os.path.abspath(__file__))

track_file = sys.argv[1] if  len(sys.argv) >= 2 else \
    os.path.join(demo_dir, 'tracks_CAM5-1-0.25degree_All-Hist_est1_v3_run2.bin' )#'tracks_1990s_3hr_mdd_4800.bin')

tex_file = sys.argv[2] if len(sys.argv) >= 3 else \
    os.path.join(demo_dir, 'earthmap4kgy.png')

year = int(sys.argv[3]) if len(sys.argv) >= 4 else 2015

box = [float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), \
    float(sys.argv[7])] if len(sys.argv) >= 8 else [180, 340.0, 0, 80.0]


# first in the pipeline is a reader to read the candidates table
reader = teca_table_reader.New()
reader.set_file_name(track_file)

# filter to remove all but the year of interest
year_filter = teca_table_remove_rows.New()
year_filter.set_input_connection(reader.get_output_port())
year_filter.set_mask_expression('!(year == %d)'%(year))

# filter by geographic area
box_filter = teca_table_remove_rows.New()
box_filter.set_input_connection(year_filter.get_output_port())
box_filter.set_mask_expression( \
    '!((lon >= %f) && (lon <= %f) && (lat >= %f) && (lat <= %f))'%( \
    box[0], box[1], box[2], box[3]))

# a 'tap' into the pipeline gives access to the data
tap = teca_dataset_capture.New()
tap.set_input_connection(box_filter.get_output_port())

# this runs the pipeline
tap.update()

# to access the dataset we need to convert from abstract
# type to concrete type here a table
track_table = as_teca_table(tap.get_dataset())
if track_table is None:
    sys.stderr.write('no data was loaded\n\n')
    sys.exit(-1)

# the table is a collection of columns, columns are accessed by
# name.
lon = track_table.get_column('lon').as_array()
lat = track_table.get_column('lat').as_array()
ids = track_table.get_column('track_id').as_array()

lon=np.load("./gt_lon.npy")
lat=np.load("./gt_lat.npy")
#ids=np.load("../pre/tid.npy")

lon_p=np.load("./pr_lon.npy")
lat_p=np.load("./pr_lat.npy")
#ids_p=np.load("../pre/tid.npy")


# ------------------------------------------------------------
# part 2
# ------------------------------------------------------------
# at this point you have access to the candidate data
# as numpy arrays in the above variables (lon, lat, psl, w, ...)
# below shows how to plot these using matplotlib
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.image as plt_img

fig = plt.figure(figsize=(8,5), dpi=150)

ext = [np.min(lon), np.max(lon), np.min(lat), np.max(lat)]
ext2 = [np.min(lon_p), np.max(lon_p), np.min(lat_p), np.max(lat_p)]

print ext,ext2


# load the blue marble texture for reference
tex = plt_img.imread(tex_file)
i0 = int(tex.shape[1]/360.0*ext[0])
i1 = int(tex.shape[1]/360.0*ext[1])
j0 = int(-((ext[3] + 90.0)/180.0 - 1.0)*tex.shape[0])
j1 = int(-((ext[2] + 90.0)/180.0 - 1.0)*tex.shape[0])
plt.imshow(tex[j0:j1, i0:i1], extent=ext, aspect='auto')

# get the track ids
uids = np.unique(ids)
n_tracks = len(uids)

# map color
cmap = plt.get_cmap('viridis')
col_ii = cmap(uids)

# plot each track
i = 0
#while i < n_tracks:
#    tid = uids[i]
#    ii = np.where(ids == tid)[0]
for ii in range(len(lon)):
    lon_ii = lon[ii]
    lat_ii = lat[ii]
    #ids_ii = ids[ii]
    lon_pp = lon_p[ii]
    lat_pp = lat_p[ii]
  #  ids_pp = ids_p[ii]
  #  ax = plt.plot(lon_ii, lat_ii, '-', c=col_ii[i], alpha=0.5, linewidth=2)
    ax = plt.scatter(lon_ii, lat_ii, c="yellow",s=7)
    ax = plt.scatter(lon_pp, lat_pp,c="red",s=4)
    i += 1

# add candidates on top
plt.xlim()
plt.ylim()
plt.grid(True)
plt.title('Tracks \n %0.2f < lon < %0.2f, %0.2f < lat < %0.2f'%( \
     box[0], box[1], box[2], box[3]), fontweight='bold')
plt.xlabel('deg lon',fontweight='bold')
plt.ylabel('deg lat',fontweight='bold')

plt.savefig('oneTC_%f_%f_%f_%f.png'%(
     box[0], box[1], box[2], box[3]))

plt.show()
