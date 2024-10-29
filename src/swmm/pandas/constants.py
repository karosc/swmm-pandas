import os.path

_this_dir, _this_filename = os.path.split(__file__)

# test_inp_path = os.path.join(_this_dir, "example_data/Model.inp")
example_out_path = os.path.join(_this_dir, "example_data/Model.out")
example_rpt_path = os.path.join(_this_dir, "example_data/Model.rpt")

# volume
gal_per_cf = 7.48052
L_per_gal = 3.78541

cf_per_gal = 1 / gal_per_cf
gal_per_L = 1 / L_per_gal

# flowrates
seconds_per_day = 24 * 60 * 60

cfs_per_mgd = cf_per_gal * 1_000_000 * seconds_per_day
lps_per_mgd = L_per_gal * 1_000_000 * seconds_per_day


mgd_per_cfs = 1 / cfs_per_mgd
mgd_per_lps = 1 / lps_per_mgd

# mass
lbs_per_kg = 2.20462
lbs_per_mg = 2.20462 / 1_000_000
lbs_per_ug = lbs_per_mg / 1_000

kg_per_lbs = 1 / lbs_per_kg
mg_per_lbs = 1 / lbs_per_mg
ug_per_lbs = 1 / lbs_per_ug

# length
mm_per_inch = 25.4
inch_per_mm = 1 / mm_per_inch
