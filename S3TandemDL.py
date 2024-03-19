#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: connornelson
"""
import requests
import pandas as pd
import subprocess
import os
import shutil
import numpy as np
from glob import glob
import datetime
from shapely.geometry import Polygon
from shapely.wkt import loads
import geopandas as gpd
import pyproj
from shapely.ops import transform
from joblib import Parallel, delayed

##############################################################################################################################################################
#####################################################  Functions  ############################################################################################
##############################################################################################################################################################

def get_access_token(username, password):
        p =  subprocess.run(f"curl --location --request POST 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token' \
                --header 'Content-Type: application/x-www-form-urlencoded' \
                --data-urlencode 'grant_type=password' \
                --data-urlencode 'username={username}' \
                --data-urlencode 'password={password}' \
                --data-urlencode 'client_id=cdse-public'", shell=True,capture_output=True, text=True)
        token = p.stdout.split('"access_token":')[1].split('"')[1]
        return token

#=============================================================================================================================================================#

def in_arctic_filter(search_results_df):
    arctic_polygon = Polygon([(-180, 65), (180, 65), (180, 90), (-180, 90), (-180, 65)])
    for i, row in search_results_df.iterrows():
        footprint = loads(row['Footprint'].split(';')[1])
        # search_results_df['in_arctic'] = pd.Series(dtype=bool)
        if (arctic_polygon.contains(footprint)) or (arctic_polygon.intersects(footprint)):
            search_results_df.loc[i, 'in_arctic'] = True
        else:
            search_results_df.loc[i, 'in_arctic'] = False
    search_results_df = search_results_df[search_results_df['in_arctic']]
    return search_results_df

#=============================================================================================================================================================#

def over_arctic_sea_ice_filter(search_results_df):
    search_results_df['Month'] = [row['SensingStart'].strftime('%m') for i,row in search_results_df.iterrows()] 
    source_crs = pyproj.CRS("EPSG:4326")
    target_crs = pyproj.CRS("EPSG:3411") 
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    for month, group in search_results_df.groupby('Month'):
        extent_gdf = gpd.read_file(f'./SI_extent_polygons/extent_N_2018{month}_polygon_v3.0/extent_N_2018{month}_polygon_v3.0.shp')
        for i, row in group.iterrows():
            footprint = loads(row['Footprint'].split(';')[1])
            transformed_footprint = transform(transformer.transform, footprint).buffer(0.1)
            # search_results_df['sea_ice'] = pd.Series(dtype=bool)
            if (len(extent_gdf[extent_gdf.intersects(transformed_footprint)]) > 0 ) or (len(extent_gdf[extent_gdf.contains(transformed_footprint)]) > 0):
                search_results_df.loc[i, 'sea_ice'] = True
            else:
                search_results_df.loc[i, 'sea_ice'] = False
    search_results_df = search_results_df[search_results_df['sea_ice']]
    return search_results_df

#=============================================================================================================================================================#

def get_olci_search_results(date):
         
    json = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-3' and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'OL_1_EFR___') and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'timeliness' and att/OData.CSC.StringAttribute/Value eq 'NT') and ContentDate/Start gt {date.strftime('%Y-%m-%dT%H:%M:%SZ')} and ContentDate/End lt {(date+pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')}&$top=1000").json()

    search_results_df = pd.DataFrame.from_dict(json['value'])
    search_results_df['SensingStart'] = [pd.to_datetime(row['ContentDate']['Start']) for i,row in search_results_df.iterrows()] 
    search_results_df.sort_values(by='SensingStart', inplace=True)

    # in_arctic_filter(search_results_df)
    search_results_df = over_arctic_sea_ice_filter(search_results_df)

    return search_results_df

#=============================================================================================================================================================#

def download_OLCI_from_df(results_df_row, cop_dspace_usrnm, cop_dspace_psswrd):

    token = get_access_token(cop_dspace_usrnm, cop_dspace_psswrd)

    save_loc = f"./S3Tandem/OLCI/{results_df_row['SensingStart'].strftime('%Y')}/{results_df_row['SensingStart'].strftime('%m')}/{results_df_row['Name'][:-5]}"
    os.makedirs(save_loc, exist_ok=True)
    zip_save_name = results_df_row['Name'][:-5]+'.zip'
    zip_path = os.path.join(save_loc,zip_save_name)

    # Download the desired product
    print(f'Downloading {zip_save_name}')
    dl_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({results_df_row['Id']})/$value"
    dl = subprocess.run(f"curl -H 'Authorization: Bearer {token}' -o {zip_path} --location-trusted '{dl_url}'", shell=True)
    # This web server doesn't support resume download requests, so we will have to download incomplete donloads from scratch
    if dl.returncode == 33:
        print("Reattempting to download product...")
        dl = subprocess.run(f"curl -H 'Authorization: Bearer {token}' -o {zip_path} --location-trusted '{dl_url}'", shell=True)

    shutil.unpack_archive(zip_path, save_loc, 'zip') 
    print(f"{zip_save_name} file unpacked successfully.")
    os.remove(zip_path)

#=============================================================================================================================================================#
#=============================================================================================================================================================#
#=============================================================================================================================================================#

parent_dir = f"/Users/weibinchen/Desktop/UCL/PhD_Year_1/Data/OLCI_Data/" #The path of parent directory for all this

# DATA SEARCH (Based on https://documentation.dataspace.copernicus.eu/APIs/OData.html)
# Define search parameters


start_date = pd.to_datetime('2018-06-14 05:00:00') 
end_date = pd.to_datetime('2018-06-14 17:00:00')

results_list = Parallel(n_jobs=-1)(delayed(get_olci_search_results)(date) for date in pd.date_range(start_date,end_date))
results_df = pd.concat(results_list).reset_index(drop=True)


# PRODUCT DOWNLOAD (Based on https://documentation.dataspace.copernicus.eu/APIs/OData.html)
# Get authorisation token (have to make account with Copernicus Data Space https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/auth?client_id=cdse-public&response_type=code&scope=openid&redirect_uri=https%3A//dataspace.copernicus.eu/%3Fconfirmed%3D1)
cop_dspace_usrnm = 'ucfbwc0@ucl.ac.uk' # Your copernicus dataspace username (make account @ https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/auth?client_id=cdse-public&response_type=code&scope=openid&redirect_uri=https%3A//dataspace.copernicus.eu/%3Fconfirmed%3D1)
cop_dspace_psswrd = '' # Your copernicus dataspace password

# Download the first few pairs to test
test_results_df = results_df.iloc[:6]
# The max simulataneous download sessions allowed is 4
Parallel(n_jobs=4)(delayed(download_OLCI_from_df)(row,cop_dspace_usrnm,cop_dspace_psswrd) for i, row in test_results_df.iterrows())
