# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:06:38 2021

@author: Derrick Muheki
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import gridspec
from scipy import stats
from scipy.special import logsumexp
import seaborn as sns
from numpy.linalg import inv, det
from statistics import mean
from fractions import Fraction




#%% SETTING UP THE CURRENT WORKING DIRECTORY; for both the input and output folders
cwd = os.getcwd()


#%% Function to extract starting year from file name to use in the next function for the start period of the data
def read_start_year(file):
    """ Read the starting year of the data from the file name


    Parameters
    ----------
    file : files in data directory (string)

    Returns
    -------
    start_year (integer)

    """
    # find the first consecutive 4 digits in the file name; which in our case represents the starting year of the data
    years = re.findall('\\d{4}', file) 
    start_year = int(years.pop(0))
    return start_year  #returns start year that is used in decoding the times for the xarray data


#%% Function for reading NetCDF4 data 
def nc_read(file, start_year, type_of_extreme_climate_event, time_dim):
    
    """ Reading netcdfs based on occurrence variable & Clip out data for the study area: The East African Region 
    
    Parameters
    ----------
    file : files in data directory (string)
    occurrence_variable : target variable name (string)
    
    Returns
    ------- 
    Xarray data array
    """
    # initiate as dataset 
    ds = xr.open_dataset(file,decode_times=False)
    
    # convert to data array for target variable
    da = ds['exposure']
    
    if time_dim:
        # manually decode times from integer to datetime series 
        new_dates = pd.date_range(start=str(start_year)+'-1-1', periods=da.sizes['time'], freq='YS')
        da['time'] = new_dates
    
    
    # Clip out data for East African Region that lies between LATITUDES 24N and 13S & LONGITUDES 18E and 55E
    latitude_bounds, longitude_bounds =[23.75, -12.75], [18.25, 54.75]
    clipped_da = da.sel(lat=slice(*latitude_bounds), lon=slice(*longitude_bounds)) #Clipped dataset
       
    return clipped_da


#%% Function for returning array showing the occurence of an extreme climate event at a location per year
def extreme_event(extreme_event_data):
    """ Investigates the occurrence of an extreme event in a grid (location) per year
    
    Parameters
    ----------
    extreme_event_data : Xarray data array

    Returns
    Xarray data array (boolean where 1 means the extreme event was recorded in that location during that year)

    """
    
    extreme_event = xr.where(extreme_event_data>0.005, 1, extreme_event_data) #returns 1 where extreme event was recorded in that location during that year
    
    return extreme_event


#%% Select from a list of Global Impact Models e.g. GHMs, GGCMS, GVMs etc. for which the Global Climate Models are based on
def impact_model(extreme_event, gcm):
    """ Select from a list of Global Impact Models e.g. GHMs, GGCMS, GVMs etc. driven by the same Global Climate Models (GCMs) 
    
    Parameters
    ----------
    extreme_event: String
    gcm: String    

    Returns
    -------
    filtered_dataset: Global impact model' files driven by the same Global Climate Model and directories for the files (dataset).

    """
    
    # List of Global Impact Models e.g. GHMs, GGCMS, GVMs etc.
    list_of_gms = os.path.join(cwd, extreme_event)
    
    all_available_of_impact_model_data_files_under_gcm = []  # List of available impact models per extreme event category
    
    
    for gm in os.listdir(list_of_gms):       
        gms = os.path.join(list_of_gms, gm)
        for file in os.listdir(gms):
            impact_model_files_directory = os.path.join(gms, file)
            all_available_of_impact_model_data_files_under_gcm.append(impact_model_files_directory)
        
    # Filter out datasets to include only impact models driven by the same GCM 
    filter_data_with_gcm = re.compile('.*{}*'.format(gcm))
    filtered_dataset = list(filter(filter_data_with_gcm.match,all_available_of_impact_model_data_files_under_gcm))
    
    #print(filtered_dataset)  
    
    return filtered_dataset


#%% Function for returning extreme event occurrence per grid for a given scenario and given time period considering an ensemble of GCMs
def extreme_event_occurrence(type_of_extreme_climate_event, list_of_impact_models_driven_by_same_gcm, scenario_of_dataset):
    """ Function for returning extreme event occurrence per grid for a given scenario and given time period considering an ensemble of GCMs
    
    Parameters
    ----------
    type_of_extreme_climate_event : String
    list_of_impact_models_driven_by_same_gcm : List
    scenario_of dataset : String

    Returns
    -------
    Tuple (with [0] & [1] Xarray (Extreme event occurrences for respective time period))

    """

    # Creating list for all available data files according to selected criteria: Extreme Event Type, Impact Model, Global Climate Model and Time Period/Scenario/RCP
    extreme_event_datasets_in_the_scenario =[] 
    for gcm in list_of_impact_models_driven_by_same_gcm:
        if('landarea' in gcm): #Selecting only the land area datafiles (exposure on land), thus excluding the available population datafiles within the dataset
            if(scenario_of_dataset in gcm): #Selecting data for only the selected time period/scenario/RCP
                #data_files = os.path.join(impact_model_list_of_gcms[1], gcm)
                extreme_event_datasets_in_the_scenario.append(gcm)
            
    print('The following files from their respective Global Climate Models (GCMs) are available for the time scenario and Global Impact Model selected: \n \n ')
    print(*extreme_event_datasets_in_the_scenario, sep='\n')
    print('------------------------------------------------------------------------------------------------------------------------------------')
    print('')
    print('*********PROCESSING DATA************PLEASE WAIT*********** \n')

    # Filtering the extreme event datasets in the scenario for the different time periods (1661-1860, 1861-2005, 2006-2099 and 2100-2299)

    # Filter out datasets for the period from 1661 until 1860    
    #filter_data_from_1661_until_1860 = re.compile('.*_1661_*') # start year of the period = 1661
    #extreme_event_data_from_1661_until_1860 =list(filter(filter_data_from_1661_until_1860.match,extreme_event_datasets_in_the_scenario))

    # Filter out datasets for the period from 1861 until 2005  
    filter_data_from_1861_until_2005 = re.compile('.*_1861_*') # start year of the period = 1861
    extreme_event_data_from_1861_until_2005 =list(filter(filter_data_from_1861_until_2005.match,extreme_event_datasets_in_the_scenario))

    # Filter out datasets for the period from 2006 until 2099
    filter_data_from_2006_until_2099 = re.compile('.*_2006_*') # start year of the period = 2006
    extreme_event_data_from_2006_until_2099 =list(filter(filter_data_from_2006_until_2099.match,extreme_event_datasets_in_the_scenario))

    # Filter out datasets for the period from 2100 until 2299
    # filter_data_from_2100_until_2299 = re.compile('.*_2100_*') # start year of the period = 2100
    # extreme_event_data_from_2100_until_2299 =list(filter(filter_data_from_2100_until_2299.match,extreme_event_datasets_in_the_scenario))


    # =============================================================================
    # OCCURRENCE OF EXTREME EVENT WITHIN SCENARIO CONSIDERING THE MAXIMUM/EXTREME VALUES PER GRID FROM AN ENSEMBLE OF GCMS
    # =============================================================================

    # OCCURRENCE OF EXTREME EVENT FROM 1861 UNTIL 2005

    occurrence_of_extreme_event_datasets_from_1861_until_2005 =[]
    for file in extreme_event_data_from_1861_until_2005:
        start_year_of_the_data = read_start_year(file) # function to get the starting year of the data from file name
        extreme_event_from_1861_until_2005 = nc_read(file, start_year_of_the_data,type_of_extreme_climate_event, time_dim=True) # function to read the NetCDF4 files based on occurrence variable
    
        # occurence of an extreme event...as a boolean...true or false
        occurrence_of_extreme_event_from_1861_until_2005 = extreme_event(extreme_event_from_1861_until_2005) #returns 1 where an extreme event was recorded in that location during that year
    
        # add the array with occurrences per GCM for same time period to list (that shall be iterated to get the extremes from the ENSEMBLE of these different GCMs)
        occurrence_of_extreme_event_datasets_from_1861_until_2005.append(occurrence_of_extreme_event_from_1861_until_2005)

    if len(occurrence_of_extreme_event_datasets_from_1861_until_2005) == 0: # check for empty data set list
        print('No data available for the selected extreme events for the selected GCM scenario during the period from 1861 until 2015 \n *********PROCESSING DATA************PLEASE WAIT*********** \n')
        occurrence_of_extreme_event_considering_ensemble_of_gcms_from_1861_until_2005 = xr.DataArray([]) #create empty array
    else: # occurrence of extreme event in arrays for each available impact model driven by the same GCM
        occurrence_of_extreme_event_considering_ensemble_of_gcms_from_1861_until_2005 = occurrence_of_extreme_event_datasets_from_1861_until_2005


    # OCCURRENCE OF EXTREME EVENT FROM 2006 UNTIL 2099

    occurrence_of_extreme_event_datasets_from_2006_until_2099 =[]
    for file in extreme_event_data_from_2006_until_2099:
        start_year_of_the_data = read_start_year(file) # function to get the starting year of the data from file name
        extreme_event_from_2006_until_2099 = nc_read(file, start_year_of_the_data, type_of_extreme_climate_event, time_dim=True) # function to read the NetCDF4 files based on occurrence variable
    
        # occurence of an extreme event...as a boolean...true or false
        occurrence_of_extreme_event_from_2006_until_2099 = extreme_event(extreme_event_from_2006_until_2099) #returns 1 where an extreme event was recorded in that location during that year
    
        # add the array with occurrences per GCM for ame time period to list (that shall be iterated to get the extremes from the ENSEMBLE of these different GCMs)
        occurrence_of_extreme_event_datasets_from_2006_until_2099.append(occurrence_of_extreme_event_from_2006_until_2099)

    if len(occurrence_of_extreme_event_datasets_from_2006_until_2099) == 0: # check for empty data set list
        print('No data available for the selected extreme events for the selected GCM scenario during the period from 2006 until 2099 \n *********PROCESSING DATA************PLEASE WAIT*********** \n')
        occurrence_of_extreme_event_considering_ensemble_of_gcms_from_2006_until_2099 = xr.DataArray([]) #create empty array
    else: # occurrence of extreme event in arrays for each available impact model driven by the same GCM
        occurrence_of_extreme_event_considering_ensemble_of_gcms_from_2006_until_2099 = occurrence_of_extreme_event_datasets_from_2006_until_2099

    
    print('\n *******************PROCESSING************************ \n')
    
    return occurrence_of_extreme_event_considering_ensemble_of_gcms_from_1861_until_2005, occurrence_of_extreme_event_considering_ensemble_of_gcms_from_2006_until_2099


#%% Function for returning array showing locations with the occurrence of compound events within the same location in the same year
def compound_event_occurrence(occurrence_of_event_1, occurrence_of_event_2):
    
    """ Compare two arrays with occurrence of extreme climate events at the same locations and during the same years
    
    Parameters
    ----------
    occurrence_of_event_1 & occurrence_of_event_2 : Xarray data arrays for occurrence of extreme events
    
    Returns
    -------
    Xarray data array (boolean with true for locations with the occurrence of both events within the same year)
    """
       
    compound_event = np.logical_and(occurrence_of_event_1==1, occurrence_of_event_2==1)
    compound_bin  = xr.where(compound_event == True, 1, 0)#returns True for locations with occurence of compound events within same location in same year
    nanq = np.logical_and(np.isnan(occurrence_of_event_1),np.isnan(occurrence_of_event_2)) #returns array where nan is true for both event 1 and event 2 
    compound_event_bin = xr.where(nanq==True, np.nan, compound_bin) #returns 1 where extreme event was recorded in that location during that year
    
       
    return compound_event_bin


#%% Function for changing event names from the original Lange et. al (2020) dataset folder names

def event_name(type_of_extreme_climate_event_selected):
    """ changing event names from the original Lange et. al (2020) dataset folder names
    
    Parameters
    ----------
    type_of_extreme_climate_event_selected : String

    Returns
    -------
    String (Extreme Event name)

    """

    if type_of_extreme_climate_event_selected == 'floodedarea':
        event_name = 'River Floods'
    if type_of_extreme_climate_event_selected == 'driedarea':
        event_name = 'Droughts'
    if type_of_extreme_climate_event_selected == 'heatwavedarea':
        event_name = 'Heatwaves'
    if type_of_extreme_climate_event_selected == 'cropfailedarea':
        event_name = 'Crop Failures'
    if type_of_extreme_climate_event_selected =='burntarea':
        event_name = 'Wildfires'
    if type_of_extreme_climate_event_selected == 'tropicalcyclonedarea':
        event_name ='Tropical Cyclones'
    
    return event_name


#%% Function for calculating the total number of years per location that experienced compound events. 
def total_no_of_years_with_compound_event_occurrence(occurrence_of_compound_event):
    
    """ Determine the total number of years throught the data period per location for which a compound extreme event was experienced
    
    Parameters
    ----------
    occurrence of compound extreme event : Xarray data array (boolean with true for locations with the occurrence of both events within the same year)
    
    Returns
    -------
    Xarray with the total number of years throught the data period per location for which a compound extreme event was experienced   
    """
    
    # Number of years per region that experienced compound events
    no_of_years_with_compound_events = (occurrence_of_compound_event).sum('time',skipna=False)
   
    return no_of_years_with_compound_events  


#%% Function for plotting map showing the total number of years per location that experienced compound events. FOR VISUALIZATION
def plot_total_no_of_years_with_compound_event_occurrence(no_of_years_with_compound_events, event_1_name, event_2_name, time_period, gcm, scenario):
    
    """ Plot a map showing the total number of years throught the data period per location for which a compound extreme event was experienced
    
    Parameters
    ----------
    occurrence of compound extreme event : Xarray data array (boolean with true for locations with the occurrence of both events within the same year)
    event_1_name, event_2_name : String (Extreme Events)
    gcm: String (Driving GCM)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the total number of years throught the data period per location for which a compound extreme event was experienced   
    """
       
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
          
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    
    # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
   
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = False, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Plot number of years with occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(no_of_years_with_compound_events, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = plt.cm.get_cmap('viridis', 12))
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.55,0.9,'{}, {}'.format(time_period, scenario), fontsize = 8)
    plt.gcf().text(0.25,0.03,'{}'.format(gcm), fontsize= 6)
    
    # Add the title and legend to the figure and show the figure
    plt.title('Occurrence of Compound {} and {} \n'.format(event_1_name, event_2_name),fontsize=10) #Plot title  
    
    # discrete color bar legend
    #bounds = [0,5,10,15,20,25,30]
    plt.clim(0,30)
    plt.colorbar(orientation = 'vertical', extend = 'max').set_label(label = 'Number of years', size = 9) #Plots the legend color bar
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    plt.show()
    
    #plt.close()
    
    return no_of_years_with_compound_events
    

#%% Function for calculating the Probability of joint occurrence of the extreme events in one grid cell over the entire dataset period
def probability_of_occurrence_of_compound_events(no_of_years_with_compound_events, occurrence_of_compound_event):
    
    """ Determine the probability of occurrence of a compound extreme climate event over the entire dataset time period
    
    Parameters
    ----------
    occurrence of compound extreme event : Xarray data array (boolean with true for locations with the occurrence of both events within the same year)
    
    Returns
    -------
    Xarray with the probability of joint occurrence of two extreme climate events over the entire dataset time period
    
    """
    # Total number of years in the dataset
    total_no_of_years_in_dataset = len(occurrence_of_compound_event)
    
    # Probability of occurrence
    probability_of_occurrence_of_the_compound_event = no_of_years_with_compound_events/total_no_of_years_in_dataset
    
    return probability_of_occurrence_of_the_compound_event


#%% Function for plotting map showing the Probability of joint occurrence of the extreme events in one grid cell over the entire dataset period
def plot_probability_of_occurrence_of_compound_events(no_of_years_with_compound_events, event_1_name, event_2_name, time_period, gcm, scenario):
    
    """ Plot a map showing the probability of occurrence of a compound extreme climate event over the entire dataset time period
    
    Parameters
    ----------
    occurrence of compound extreme event : Xarray data array (boolean with true for locations with the occurrence of both events within the same year)
    event_1_name, event_2_name : String (Extreme Events)
    gcm : String (Driving GCM)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the probability of joint occurrence of two extreme climate events over the entire dataset time period
    
    """
    
    # Probability of occurrence
    probability_of_occurrence_of_the_compound_event = no_of_years_with_compound_events/50  # as 50 years were considered to determine the probability 
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = True, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(probability_of_occurrence_of_the_compound_event, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = plt.cm.get_cmap('viridis', 10))
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.50,0.9,'{}, {}'.format(time_period, scenario), fontsize = 10)
    plt.gcf().text(0.25,0.03,'{}'.format(gcm), fontsize= 10)
    
    # Add the title and legend to the figure and show the figure
    plt.title('Probability of joint occurrence of {} and {} \n'.format(event_1_name, event_2_name),fontsize=11) #Plot title
    
    # discrete color bar legend
    #bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.colorbar( orientation = 'vertical').set_label(label = 'Probability of joint occurrence', size = 11) #Plots the legend color bar
    plt.clim(0,1)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    plt.show()
    
    #plt.close()
    
    return probability_of_occurrence_of_the_compound_event


#%% Function for plotting map showing the Probability Ration (PR) of joint occurrence of the extreme events in one grid cell over the entire dataset period
def plot_probability_ratio_of_occurrence_of_compound_events(probability_of_occurrence_of_the_compound_event_in_scenario, probability_of_occurrence_of_the_compound_event_in_historical_early_industrial_scenario, event_1_name, event_2_name, time_period, gcm, scenario):
    
    """ Plot a map showing the probability of occurrence of a compound extreme climate event over the entire dataset time period
    
    Parameters
    ----------
    probability_of_occurrence_of_the_compound_event_in_scenario : Xarray data array (probability of joint occurrence of two extreme climate events over the entire dataset time period in new scenario)
    probability_of_occurrence_of_the_compound_event_in_historical_early_industrial_scenario : Xarray data array (probability of joint occurrence of two extreme climate events over the entire dataset time period in historical early industrial time period)
    event_1_name, event_2_name : String (Extreme Events)
    gcm : String (Driving GCM)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the Probability Ratio (PR) of joint occurrence of two extreme climate events to compare the change in new scenario from the historical early industrial times
    
    """
    
    # Probability Ratio (PR) of occurrence: ratio between probability of occurrence of compound event in new situation (scenario) to the probability of occurrence in the reference situtaion (historical early industrial times)
    probability_of_ratio_of_occurrence_of_the_compound_event = probability_of_occurrence_of_the_compound_event_in_scenario/probability_of_occurrence_of_the_compound_event_in_historical_early_industrial_scenario
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = True, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(probability_of_ratio_of_occurrence_of_the_compound_event, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = plt.cm.get_cmap('viridis', 12))
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.23,0.9,'Comparing {} ({}) to 1861-1890 (historical)'.format(time_period, scenario), fontsize = 9)
    plt.gcf().text(0.25,0.03,'{}'.format(gcm), fontsize= 9)
    
    # Add the title and legend to the figure and show the figure
    plt.title('Average Probability Ratio of Joint Occurrence  of {} and {} events \n'.format(event_1_name, event_2_name),fontsize=10) #Plot title
    
    # discrete color bar legend
    #bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.colorbar( orientation = 'vertical', extend = 'max').set_label(label = 'Probability Ratio', size = 9) #Plots the legend color bar
    plt.clim(0,30)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    plt.show()
    
    #plt.close()
    
    return probability_of_ratio_of_occurrence_of_the_compound_event

#%% Plot average probability of compound events accross all impact models and all their driving GCMs
def plot_average_probability_of_occurrence_of_compound_events(average_probability_of_occurrence_of_compound_events_across_the_gcms, event_1_name, event_2_name, time_period, scenario):
    
    """ Plot a map showing the probability of occurrence of a compound extreme climate event over the entire dataset time period
    
    Parameters
    ----------
    occurrence of compound extreme event : Xarray data array (boolean with true for locations with the occurrence of both events within the same year)
    event_1_name, event_2_name : String (Extreme Events)
    gcm : String (Driving GCM)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the probability of joint occurrence of two extreme climate events over the entire dataset time period
    
    """
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = True, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(average_probability_of_occurrence_of_compound_events_across_the_gcms, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = plt.cm.get_cmap('viridis', 10))
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.252,0.9,'{}, {}'.format(time_period, scenario), fontsize = 8)
    
    # Incase you want to add the title to the plot
    #plt.title('Average probability of joint occurrence of {} and {} considering all impact models and their respective driving GCMs'.format(event_1_name, event_2_name),fontsize=11) #Plot title
    
    # discrete color bar legend
    #bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.colorbar( orientation = 'vertical').set_label(label = 'Probability of joint occurrence', size = 11) #Plots the legend color bar
    plt.clim(0,1)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    
    # Change this directory to save the plots to your desired directory
    #plt.savefig('C:/Users/dmuheki/OneDrive/PhD/Masters_thesis_paper/Ongoing_results/Average probability of joint occurrence of {} and {} under the {} scenario considering all impact models and their respective driving GCMs.pdf'.format(event_1_name, event_2_name, scenario), dpi = 300)
     
    plt.show()
    #plt.close()
    
    return average_probability_of_occurrence_of_compound_events_across_the_gcms

#%% Function for plotting map showing the maximum number of years with consecutive compound events in the same location during the entire dataset period
def maximum_no_of_years_with_consecutive_compound_events(occurrence_of_compound_event):
    
    """Determine the maximum number of years with consecutive compound extreme events in the same location over the entire dataset period
    
    Parameters
    ----------
    occurrence of compound extreme event : Xarray data array (boolean with true for locations with the occurrence of both events within the same year)
    
    Returns
    -------
    Xarray with the maximum number of years with consecutive compound extreme events in the same location over the entire dataset period
              
    """
    
    # Calculates the cumulative sum and whenever a false is met, it resets the sum to zero. thus the .max() returns the maximum cummumlative sum along the entire time dimension
    maximum_no_of_years_with_consecutive_compound_events = (occurrence_of_compound_event.cumsum('time',skipna=False) - occurrence_of_compound_event.cumsum('time',skipna=False).where(occurrence_of_compound_event.values == False).ffill('time').fillna(0)).max('time')
       
    
    return maximum_no_of_years_with_consecutive_compound_events

#%% Function for plotting map showing the maximum number of years with consecutive compound events in the same location during the entire dataset period
def plot_maximum_no_of_years_with_consecutive_compound_events(average_maximum_no_of_years_with_consecutive_compound_events, event_1_name, event_2_name, time_period, gcm, scenario):
    
    """Plot a map showing the maximum number of years with consecutive compound extreme events in the same location over the entire dataset period
    
    Parameters
    ----------
    average_maximum_no_of_years_with_consecutive_compound_events : Xarray data array (with the maximum number of years with consecutive compound extreme events in the same location over the entire dataset period)
    event_1_name, event_2_name : String (Extreme Events)
    gcm : String (Driving GCM)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the maximum number of years with consecutive compound extreme events in the same location over the entire dataset period
              
    """
     
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    # grid= ax.gridlines(draw_labels = True, dms = True )
    # grid.top_labels = False #Removes the grid labels from the top of the plot
    # grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(average_maximum_no_of_years_with_consecutive_compound_events, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = plt.cm.get_cmap('YlOrRd', 10))
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.252,0.9,'{}, {}'.format(time_period, scenario), fontsize = 10)
    plt.gcf().text(0.25,0.03,'{}'.format(gcm), fontsize= 10)
    
    # Add the title and legend to the figure and show the figure
    plt.title('Maximum no. of consecutive years with Joint occurrence of \n{} and {} \n '.format(event_1_name, event_2_name),fontsize=11) #Plot title
    
    # discrete color bar legend
    #bounds = [0,5,10,15,20,25,30]
    
    plt.colorbar(orientation = 'vertical').set_label(label = 'Number of years', size = 11) #Plots the legend color bar
    plt.clim(0,50)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels

    plt.show()
    plt.close()
    
    return average_maximum_no_of_years_with_consecutive_compound_events


#%% Function for plotting map showing the averAGE maximum number of years with consecutive compound events in the same location during the entire dataset period CONSIDERING ALL IMPACT MODELS AND THIER RESPECTIVE DRIVING GCMS
def plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events, event_1_name, event_2_name, time_period, scenario):
    
    """Plot a map showing the maximum number of years with consecutive compound extreme events in the same location over the entire dataset period
    
    Parameters
    ----------
    average_max_no_of_consecutive_years_with_compound_events : Xarray data array (with the maximum number of years with consecutive compound extreme events in the same location over the entire dataset period)
    event_1_name, event_2_name : String (Extreme Events)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the average maximum number of years with consecutive compound extreme events in the same location over the entire dataset period
              
    """
    
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    # grid= ax.gridlines(draw_labels = True, dms = True )
    # grid.top_labels = False #Removes the grid labels from the top of the plot
    # grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(average_max_no_of_consecutive_years_with_compound_events, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = plt.cm.get_cmap('YlOrRd', 12))
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.252,0.9,'{}, {}'.format(time_period, scenario), fontsize = 10)
    
    # Incase you want to add the title and legend to the figure and show the figure
    #plt.title('Maximum no. of consecutive years with Joint occurrence of \n{} and {} \n '.format(event_1_name, event_2_name),fontsize=11) #Plot title
    
    # discrete color bar legend
    #bounds = [0,5,10,15,20,25,30]
    
    plt.colorbar(orientation = 'vertical', extend = 'max').set_label(label = 'Number of years', size = 10) #Plots the legend color bar
    plt.clim(0,30)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels

    
    # Change this directory to save the plots to your desired directory
    #plt.savefig('C:/Users/dmuheki/OneDrive/PhD/Masters_thesis_paper/Ongoing_results/Average maximum number of years with concurrent {} and {} demonstrated by multi model ensembles under the {} scenario.pdf'.format(event_1_name, event_2_name, scenario), dpi = 300)
    
    plt.show()
    #plt.close()
    
    return average_max_no_of_consecutive_years_with_compound_events



#%% A timeseries showing the fraction of the area affected by an extreme event across full timescale in the scenario
def timeseries_fraction_of_area_affected(occurrence_of_extreme_event, entire_globe_grid_cell_areas_in_xarray):
    """ A timeseries showing the fraction of the area affected (AS A PERCENTAGE) by an extreme event OR compound extreme event

    Parameters
    ----------
    occurrence_of_extreme_event : Xarray

    Returns
    -------
    Timeseries (Xarray)

    """
    
    
    # Clip out cell grid area data for East African Region that lies between LATITUDES 24N and 13S & LONGITUDES 18E and 55E
    latitude_bounds, longitude_bounds =[23.75, -12.75], [18.25, 54.75]
    east_africa_grid_cell_areas = entire_globe_grid_cell_areas_in_xarray.sel(lat=slice(*latitude_bounds), lon=slice(*longitude_bounds)) #Clipped dataset
    
    
    # mask area with nan values e.g. over the ocean
    masked_area_affected_by_occurrence_of_extreme_event = xr.where(np.isnan(occurrence_of_extreme_event), np.nan, east_africa_grid_cell_areas)
    # mask area without occurrence of compound events
    area_affected_by_occurrence_of_extreme_event = xr.where(occurrence_of_extreme_event == 1, masked_area_affected_by_occurrence_of_extreme_event, np.nan)
    # total area affected by occurrence of compound events per year
    total_area_affected_by_occurrence_of_extreme_event = area_affected_by_occurrence_of_extreme_event.sum(dim=['lon', 'lat'], skipna=True)
    
    # total area of the region (excluding the ocean)
    total_area_of_the_region = masked_area_affected_by_occurrence_of_extreme_event.sum(dim=['lon','lat'], skipna= True)
    
    # percentage of area affected
    percentage_of_area_affected_by_occurrence_of_extreme_event = (total_area_affected_by_occurrence_of_extreme_event/total_area_of_the_region)*100
     
    
    return percentage_of_area_affected_by_occurrence_of_extreme_event
            
#%% Plot timeseries showing the fraction of the area affected by the compound extreme event across full timescale in the scenario
def plot_timeseries_fraction_of_area_affected_by_compound_events(timeseries_of_joint_occurrence_of_compound_events, event_1_name, event_2_name, gcm):
    """ Plot timeseries showing the fraction of the area affected by the compound extreme event across full timescale in the scenario
    
    Parameters
    ----------
    timeseries_of_joint_occurrence_of_compound_events : Tuple

    Returns
    Time series plot 
 
    """
    
    colors = [(0.996, 0.89, 0.569), (0.996, 0.769, 0.31), (0.996, 0.6, 0.001), (0.851, 0.373, 0.0549), (0.6, 0.204, 0.016)] # color palette
  
    plt.figure(figsize = (5.5,4))
    
    # Considering a 10-year moving average window to reduce the noise in the timeseries plots
    
    # historical plot #considering historical starting from 1930
    all_impact_model_historical_timeseries = xr.concat(timeseries_of_joint_occurrence_of_compound_events[0], dim='impact_model').mean(dim='impact_model', skipna= True) # Mean accross all impact models driven by the same GCM
    historical_timeseries = (all_impact_model_historical_timeseries[69:]).rolling(time=10, min_periods=1).mean()
    std_historical_timeseries = (all_impact_model_historical_timeseries[69:]).rolling(time=10, min_periods=1).std() # standard deviation accross all impact models driven by the same GCM
    lower_line_historical_timeseries = (historical_timeseries - 2 * std_historical_timeseries) # lower 95% confidence interval 
    upper_line_historical_timeseries = (historical_timeseries + 2 * std_historical_timeseries) # upper 95% confidence interval
    plot1 = historical_timeseries.plot.line(x ='time', color=(0.996, 0.89, 0.569), add_legend='False') # plot line
    plot1_fill = plt.fill_between(np.ravel(np.array((historical_timeseries.time), dtype = 'datetime64[ns]')), xr.where((lower_line_historical_timeseries.squeeze())<0,0,(lower_line_historical_timeseries.squeeze())), xr.where((upper_line_historical_timeseries.squeeze())>100,100,(upper_line_historical_timeseries.squeeze())), color=(0.996, 0.89, 0.569), alpha = 0.1) # plot the uncertainty bands #np.ravel() to avoid arrays in arrays e.g [[]]
    
    # rcp26 plot
    all_impact_model_rcp26_timeseries = xr.concat(timeseries_of_joint_occurrence_of_compound_events[1], dim='impact_model').mean(dim='impact_model', skipna= True) # Mean accross all impact models driven by the same GCM
    rcp26_timeseries = (all_impact_model_rcp26_timeseries).rolling(time=10, min_periods=1).mean()
    std_rcp26_timeseries = (all_impact_model_rcp26_timeseries).rolling(time=10, min_periods=1).std() # standard deviation accross all impact models driven by the same GCM
    lower_line_rcp26_timeseries = (rcp26_timeseries - 2 * std_rcp26_timeseries) # lower 95% confidence interval
    upper_line_rcp26_timeseries = (rcp26_timeseries + 2 * std_rcp26_timeseries) # upper 95% confidence interval
    plot2 = rcp26_timeseries.plot.line(x = 'time', color=(0.996, 0.6, 0.001), add_legend='False') # plot line
    plot2_fill = plt.fill_between(np.ravel(np.array((rcp26_timeseries.time), dtype = 'datetime64[ns]')), xr.where((lower_line_rcp26_timeseries.squeeze())<0,0,(lower_line_rcp26_timeseries.squeeze())), xr.where((upper_line_rcp26_timeseries.squeeze())>100,100,(upper_line_rcp26_timeseries.squeeze())), color=(0.996, 0.6, 0.001), alpha = 0.1) # plot the uncertainty bands
    
    # rcp60 plot
    all_impact_model_rcp60_timeseries = xr.concat(timeseries_of_joint_occurrence_of_compound_events[2], dim='impact_model').mean(dim='impact_model', skipna= True) # Mean accross all impact models driven by the same GCM
    rcp60_timeseries = (all_impact_model_rcp60_timeseries).rolling(time=10, min_periods=1).mean()
    std_rcp60_timeseries = (all_impact_model_rcp60_timeseries).rolling(time=10, min_periods=1).std() # standard deviation accross all impact models driven by the same GCM
    lower_line_rcp60_timeseries = (rcp60_timeseries - 2 * std_rcp60_timeseries) # lower 95% confidence interval
    upper_line_rcp60_timeseries = (rcp60_timeseries + 2 * std_rcp60_timeseries) # upper 95% confidence interval
    plot3 = rcp60_timeseries.plot.line(x= 'time', color= (0.851, 0.373, 0.0549), add_legend='False') # plot line
    plot3_fill = plt.fill_between(np.ravel(np.array((rcp60_timeseries.time), dtype = 'datetime64[ns]')), xr.where((lower_line_rcp60_timeseries.squeeze())<0,0,(lower_line_rcp60_timeseries.squeeze())), xr.where((upper_line_rcp60_timeseries.squeeze())>100,100,(upper_line_rcp60_timeseries.squeeze())), color=(0.851, 0.373, 0.0549), alpha = 0.1) # plot the uncertainty bands
    
    
    # legend in order of plots: historical, rcp2.6, rcp6.0
    plt.legend([(plot1[0],plot1_fill), (plot2[0],plot2_fill), (plot3[0],plot3_fill)],['historical','rcp2.6','rcp6.0'], loc = 'upper left', frameon= False, fontsize = 12)
    
    if len(timeseries_of_joint_occurrence_of_compound_events) == 4:
        
        #rcp85 plot
        all_impact_model_rcp85_timeseries = xr.concat(timeseries_of_joint_occurrence_of_compound_events[3], dim='impact_model').mean(dim='impact_model', skipna= True) # Mean accross all impact models driven by the same GCM
        rcp85_timeseries = (all_impact_model_rcp85_timeseries).rolling(time=10, min_periods=1).mean()
        std_rcp85_timeseries = (all_impact_model_rcp85_timeseries).rolling(time=10, min_periods=1).std() # standard deviation accross all impact models driven by the same GCM
        lower_line_rcp85_timeseries = (rcp85_timeseries - (2 * std_rcp85_timeseries)) # lower 95% confidence interval
        upper_line_rcp85_timeseries = (rcp85_timeseries + (2 * std_rcp85_timeseries)) # upper 95% confidence interval
        plot4 = rcp85_timeseries.plot.line(x= 'time', color= (0.6, 0.204, 0.016), add_legend='False') # plot line
        plot4_fill = plt.fill_between(np.ravel(np.array((rcp85_timeseries.time), dtype = 'datetime64[ns]')), xr.where((lower_line_rcp85_timeseries.squeeze())<0,0,(lower_line_rcp85_timeseries.squeeze())), xr.where((upper_line_rcp85_timeseries.squeeze())>100,100,(upper_line_rcp85_timeseries.squeeze())), color=(0.6, 0.204, 0.016), alpha = 0.1) # plot the uncertainty bands
        
        
        # legend in order of plots: historical, rcp2.6, rcp6.0, rcp8.5
        plt.legend([(plot1[0],plot1_fill), (plot2[0],plot2_fill), (plot3[0],plot3_fill), (plot4[0],plot4_fill)],['historical','rcp2.6','rcp6.0', 'rcp8.5'], loc = 'upper left', frameon= False, fontsize = 12)
    
    
    # Add the title and legend to the figure and show the figure
    plt.title('Timeseries showing the fraction of region with Joint occurrence of \n {} and {} '.format(event_1_name, event_2_name),fontsize=12) #Plot title
    plt.ylim(0,70)
    plt.xlabel('Years', fontsize = 10)
    plt.ylabel('Percentage of area', fontsize =10)
    plt.tight_layout()
    
    plt.gcf().text(0.12,0.03,'{}'.format(gcm), fontsize= 10)
    
    plt.show()
    
    
    return timeseries_of_joint_occurrence_of_compound_events
    

    
#%% Plot comparison timeseries showing the fraction of the area affected by extreme events across a 50 year timescale in the scenario
def plot_comparison_timeseries_fraction_of_area_affected_by_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2, timeseries_of_fraction_of_area_with_occurrence_of_compound_events_during_same_time_period, event_1_name, event_2_name, time_period, gcm, scenario):
    """ Plot comparison timeseries showing the fraction of the area affected by the extreme events across 50 year timescale in the scenario
    
    Parameters
    ----------
    timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2, timeseries_of_fraction_of_area_with_occurrence_of_compound_events_during_same_time_period : Arrays

    Returns
    Time series plot 
 
    """
    plt.figure(figsize = (5.5,4))
      
    # extreme event 1
    timeseries_of_occurrence_of_extreme_event_1.plot.line(x = 'time', color='blue', add_legend='False')
    
    # extreme event 2
    timeseries_of_occurrence_of_extreme_event_2.plot.line(x ='time', color='green', add_legend='False')
    
    # joint occurrence
    timeseries_of_fraction_of_area_with_occurrence_of_compound_events_during_same_time_period.plot.line(x ='time', color='red', add_legend='False')
    
    # legend in order of plots: historical, rcp2.6, rcp6.0
    plt.legend([event_1_name,event_2_name,'Joint occurrence'], loc = 'upper left', frameon= False)
    
    plt.title('Timeseries showing the percentage of the region with \n Occurrence of {} and {} \n '.format(event_1_name, event_2_name),fontsize=11) #Plot title
    
    plt.gcf().text(0.68,0.85,'{}, {}'.format(time_period, scenario), fontsize = 10)
    plt.gcf().text(0.12,0.03,'{}'.format(gcm), fontsize= 10)
    
    plt.ylim(0,100)
    plt.xlabel('Years', fontsize = 12)
    plt.ylabel('Percentage of area', fontsize = 12)
    plt.tight_layout()
    plot_comparison = plt.show()
    
    return plot_comparison  


#%% Function for plotting the confidence ellipse of the covariance of two extreme event occurrences
def confidence_ellipse(x, y, ax, **kwargs):
    """
    Create a plot of the covariation confidence ellipse op `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data

    Returns
    -------
    float: the Pearson Correlation Coefficient for `x` and `y`.

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties

    author : Carsten Schelp
    license: GNU General Public License v3.0 (https://github.com/CarstenSchelp/CarstenSchelp.github.io/blob/master/LICENSE)
    """
    
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    
    if det(cov) == 0:
        pearson = -9999
        #pass # avoiding a matrix with a determinant of zero (in which case inverting it would lead to an error, see: https://www.statology.org/python-numpy-linalg-singular-matrix/)
    
    else:
    
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1,1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)
    
        # Considering number of standard deviations (n_std) where: sigmas 1s, 2s or 3s represent 68%, 95% and 99.7% rule respectively (https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)
        n_std=1
    
        # calculating the stdandarddeviation of x from  the squareroot of the variance
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        
        # calculating the stdandarddeviation of y from  the squareroot of the variance
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
            
        ellipse.set_transform(transf + ax.transData)
        
# =============================================================================
#         #Plot the ellipse
#         ax.add_patch(ellipse)
# =============================================================================
        
        #plt.show()       
    return pearson


#%% Function for creating a set of the timeseries of occurrence of extremes (extreme event 1 and extreme event 2) for plotting later on the same scatter graph
def set_of_timeseries_of_occurrence_of_two_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2):
    """
    

    Parameters
    ----------
    timeseries_of_occurrence_of_extreme_event_1 : Array
    timeseries_of_occurrence_of_extreme_event_2 : Array

    Returns
    -------
    List of timeseries of occurrence of two extreme event to be used further to plot a scatter plot (extreme event 1 occurrence on x axis and extreme event 2 occurrence on y axis).

    """
    
    set_of_timeseries_of_occurrence_of_two_extreme_events= [timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2]
    
    return set_of_timeseries_of_occurrence_of_two_extreme_events

     
#%% Function for plotting the correleation of two extreme events using the Pearson correlation coefficient
def plot_correlation_with_pearson_correlation_coefficient(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events, event_1_name, event_2_name, gcm):
    """

    Parameters
    ----------
    pearson_correlation_coefficient_of_extreme_events : Tuple (containing pearson correlation coefficent - confidence ellipse patches)

    Returns
    -------
    Plot showing the correleation of two extreme events using the Pearson correlation coefficient.

    """
    
    # Setting up 4 subplots as axis objects using Gridspec:
    gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios = [1,3])
    # Add space between scatter plot and KDE plots to accomodate axis labels
    gs.update(hspace =0.1, wspace =0.1)
    
    # Set background canvas color 
    fig = plt.figure(figsize = [8,6])
    fig.patch.set_facecolor('white')
    
    ax = plt.subplot(gs[1,0]) # Scatter plot area and axis range
    ax.set_xlabel('Percentage of area affected by {}'.format(event_1_name))
    ax.set_ylabel('Percentage of area affected by {}'.format(event_2_name))
    
    axu = plt.subplot(gs[0,0], sharex = ax) # Upper KDE plot area
    axu.get_xaxis().set_visible(False) # Hide the tick marks and spines
    axu.get_yaxis().set_visible(False)
    axu.spines['right'].set_visible(False)
    axu.spines['top'].set_visible(False)
    axu.spines['left'].set_visible(False)
    
    
    axr = plt.subplot(gs[1,1], sharey = ax) # Right KDE plot area
    axr.get_xaxis().set_visible(False) # Hide the tick marks and spines
    axr.get_yaxis().set_visible(False)
    axr.spines['right'].set_visible(False)
    axr.spines['top'].set_visible(False)
    axr.spines['bottom'].set_visible(False)
        
    axl = plt.subplot(gs[0,1]) # Legend plot area
    axl.axis('off') # Hide tick marks and spines
    
       
    
    # historical plot from 1861 until 1910 
    all_values_of_pearson_from_1861_until_1910 = []
    for i in range(len(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[0])):
                  
        timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[0][i][0].squeeze()) 
        timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[0][i][1].squeeze())  

        if (np.var(timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)).values <= np.array([0.000001]):
            continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses. 
        
        pearson_from_1861_until_1910 = confidence_ellipse(timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910, timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910, ax, facecolor='none', edgecolor=(0.996, 0.89, 0.569))
        all_values_of_pearson_from_1861_until_1910.append(pearson_from_1861_until_1910)

        # Stacking the two extreme event datasets 
        combined_data= np.vstack([timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910, timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910])
        
        # Create a kernel density estimate
        kde_combined_data = stats.gaussian_kde(combined_data)
        
        # Define the grid for the contour plot
        xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
        ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
        xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
        zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Specify the number of contour lines
        num_contours = 4
        
        # Generate evenly spaced contour levels based on the data
        contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
        # Plot the contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#FEE491')
       
      
        # marginal distribution using KDE method
        kde = stats.gaussian_kde(timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)
        xx = np.linspace(timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.min(), timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.max(), 1000)
        axu.plot(xx, kde(xx), color=(0.996, 0.89, 0.569))
        kde = stats. gaussian_kde(timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)
        yy = np.linspace(timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.min(), timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.max(), 1000)
        axr.plot(kde(yy), yy, color = (0.996, 0.89, 0.569))
            
    
    average_pearson_from_1861_until_1910 = mean(all_values_of_pearson_from_1861_until_1910) # Mean Pearson's correlataion coefficient


    # historical plot from 1956 until 2005
    all_values_of_pearson_from_1956_until_2005 = []
    for i in range(len(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[1])):
            
        timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[1][i][0].squeeze())
        timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[1][i][1].squeeze())
        
        if  (np.var(timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)).values <= np.array([0.000001]):
            continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
        
        
        pearson_from_1956_until_2005 = confidence_ellipse(timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005, timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005, ax, facecolor='none', edgecolor=(0.996, 0.769, 0.31))
        all_values_of_pearson_from_1956_until_2005.append(pearson_from_1956_until_2005)
        
        
        # Stacking the two extreme event datasets 
        combined_data= np.vstack([timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005, timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005])
        
        # Create a kernel density estimate
        kde_combined_data = stats.gaussian_kde(combined_data)
        
        # Define the grid for the contour plot
        xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
        ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
        xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
        zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Specify the number of contour lines
        num_contours = 4
        
        # Generate evenly spaced contour levels based on the data
        contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
        # Plot the contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#FEC44F')
        

        #Points showing fraction affected per extreme event
        #ax.scatter(timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005, timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005, s=3, c='black')
        
        # distribution
        kde = stats.gaussian_kde(timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)
        xx = np.linspace(timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.min(), timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.max(), 1000)
        axu.plot(xx, kde(xx), color=(0.996, 0.769, 0.31))
        kde = stats. gaussian_kde(timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)
        yy = np.linspace(timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.min(), timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.max(), 1000)
        axr.plot(kde(yy), yy, color = (0.996, 0.769, 0.31))
        
     
    average_pearson_from_1956_until_2005 = mean(all_values_of_pearson_from_1956_until_2005) # Mean Pearson's correlataion coefficient
    
    
    # projected scenarios....from 2050 until 2099
    # rcp26 plot
    all_values_of_pearson_rcp26 = []
    for i in range(len(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[2])):
    
        timeseries_of_occurrence_of_extreme_event_1_rcp26 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[2][i][0].squeeze())
        timeseries_of_occurrence_of_extreme_event_2_rcp26 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[2][i][1].squeeze())
        
        if  (np.var(timeseries_of_occurrence_of_extreme_event_1_rcp26)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_rcp26)).values <= np.array([0.000001]):
            continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
        
        pearson_rcp26 = confidence_ellipse(timeseries_of_occurrence_of_extreme_event_1_rcp26, timeseries_of_occurrence_of_extreme_event_2_rcp26, ax, facecolor='none', edgecolor=(0.996, 0.6, 0.001))
        all_values_of_pearson_rcp26.append(pearson_rcp26)
        
        # Stacking the two extreme event datasets 
        combined_data= np.vstack([timeseries_of_occurrence_of_extreme_event_1_rcp26, timeseries_of_occurrence_of_extreme_event_2_rcp26])
        
        # Create a kernel density estimate
        kde_combined_data = stats.gaussian_kde(combined_data)
        
        # Define the grid for the contour plot
        xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
        ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
        xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
        zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Specify the number of contour lines
        num_contours = 4
        
        # Generate evenly spaced contour levels based on the data
        contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
        # Plot the contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#FE9900')
        
        
        #Points showing fraction affected per extreme event
        #ax.scatter(timeseries_of_occurrence_of_extreme_event_1_rcp26, timeseries_of_occurrence_of_extreme_event_2_rcp26, s=3, c='navy')
        
        # distribution
        
        
        kde = stats.gaussian_kde(timeseries_of_occurrence_of_extreme_event_1_rcp26)
        xx = np.linspace(timeseries_of_occurrence_of_extreme_event_1_rcp26.min(), timeseries_of_occurrence_of_extreme_event_1_rcp26.max(), 1000)
        axu.plot(xx, kde(xx), color=(0.996, 0.6, 0.001))
        kde = stats. gaussian_kde(timeseries_of_occurrence_of_extreme_event_2_rcp26)
        yy = np.linspace(timeseries_of_occurrence_of_extreme_event_2_rcp26.min(), timeseries_of_occurrence_of_extreme_event_2_rcp26.max(), 1000)
        axr.plot(kde(yy), yy, color = (0.996, 0.6, 0.001))
        
    
    average_pearson_rcp26 = mean(all_values_of_pearson_rcp26) # Mean Pearson's correlataion coefficient
        
    
    # rcp60 plot
    all_values_of_pearson_rcp60 = []
    for i in range(len(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[3])):
    
        timeseries_of_occurrence_of_extreme_event_1_rcp60 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[3][i][0].squeeze())
        timeseries_of_occurrence_of_extreme_event_2_rcp60 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[3][i][1].squeeze())
        
        if  (np.var(timeseries_of_occurrence_of_extreme_event_1_rcp60)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_rcp60)).values <= np.array([0.000001]):
            continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
        
        
        pearson_rcp60 = confidence_ellipse(timeseries_of_occurrence_of_extreme_event_1_rcp60, timeseries_of_occurrence_of_extreme_event_2_rcp60, ax, facecolor='none', edgecolor=(0.851, 0.373, 0.0549))
        all_values_of_pearson_rcp60.append(pearson_rcp60)
        
        # Stacking the two extreme event datasets 
        combined_data= np.vstack([timeseries_of_occurrence_of_extreme_event_1_rcp60, timeseries_of_occurrence_of_extreme_event_2_rcp60])
        
        # Create a kernel density estimate
        kde_combined_data = stats.gaussian_kde(combined_data)
        
        # Define the grid for the contour plot
        xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
        ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
        xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
        zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Specify the number of contour lines
        num_contours = 4
        
        # Generate evenly spaced contour levels based on the data
        contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
        # Plot the contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#D95F0E')
          
        
        #Points showing fraction affected per extreme event 
        #ax.scatter(timeseries_of_occurrence_of_extreme_event_1_rcp60, timeseries_of_occurrence_of_extreme_event_2_rcp60, s=3, c='red')
        
        # distribution
        kde = stats.gaussian_kde(timeseries_of_occurrence_of_extreme_event_1_rcp60)
        xx = np.linspace(timeseries_of_occurrence_of_extreme_event_1_rcp60.min(), timeseries_of_occurrence_of_extreme_event_1_rcp60.max(), 1000)
        axu.plot(xx, kde(xx), color=(0.851, 0.373, 0.0549))
        kde = stats. gaussian_kde(timeseries_of_occurrence_of_extreme_event_2_rcp60)
        yy = np.linspace(timeseries_of_occurrence_of_extreme_event_2_rcp60.min(), timeseries_of_occurrence_of_extreme_event_2_rcp60.max(), 1000)
        axr.plot(kde(yy), yy, color = (0.851, 0.373, 0.0549))
        
        
        #axl.legend(['early industrial: ' +'\u03C1'+ f'= {pearson_from_1861_until_1910:.3f}' ,'present-day: ' +'\u03C1'+ f'= {pearson_from_1956_until_2005:.3f}','rcp2.6: ' +'\u03C1'+ f'= {pearson_rcp26:.3f}', 'rcp6.0: ' +'\u03C1'+ f'= {pearson_rcp60:.3f}'], fontsize='small')
    
    average_pearson_rcp60 = mean(all_values_of_pearson_rcp60)  # Mean Pearson's correlataion coefficient  
    
    legend_elements = [Line2D([0], [0], marker ='o', color= (0.996, 0.89, 0.569), markerfacecolor=(0.996, 0.89, 0.569), label= 'early industrial: ' +'\u03C1'+ f'= {pearson_from_1861_until_1910:.3f}', markersize= 6) , Line2D([0], [0], marker ='o', color= (0.996, 0.769, 0.31), markerfacecolor=(0.996, 0.769, 0.31), label='present-day: ' +'\u03C1'+ f'= {pearson_from_1956_until_2005:.3f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.996, 0.6, 0.001), markerfacecolor=(0.996, 0.6, 0.001), label='rcp2.6: ' +'\u03C1'+ f'= {pearson_rcp26:.3f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.851, 0.373, 0.0549), markerfacecolor=(0.851, 0.373, 0.0549), label='rcp6.0: ' +'\u03C1'+ f'= {pearson_rcp60:.3f}', markersize= 6)]
    
    axl.legend(handles = legend_elements, loc = 'center', fontsize = 8, frameon=False)
    
    #if len(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events) == 5:
    if len(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[4]) != 0:
        
        # rcp85 plot
        all_values_of_pearson_rcp85 = []
        for i in range(len(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[4])):
        
            timeseries_of_occurrence_of_extreme_event_1_rcp85 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[4][i][0].squeeze())
            timeseries_of_occurrence_of_extreme_event_2_rcp85 = (gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events[4][i][1].squeeze())
            
            if  (np.var(timeseries_of_occurrence_of_extreme_event_1_rcp85)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_rcp85)).values <= np.array([0.000001]):
                continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
            
            pearson_rcp85 = confidence_ellipse(timeseries_of_occurrence_of_extreme_event_1_rcp85, timeseries_of_occurrence_of_extreme_event_2_rcp85, ax, facecolor='none', edgecolor=(0.6, 0.204, 0.016))
            all_values_of_pearson_rcp85.append(pearson_rcp85)
            
            # Stacking the two extreme event datasets 
            combined_data= np.vstack([timeseries_of_occurrence_of_extreme_event_1_rcp85, timeseries_of_occurrence_of_extreme_event_2_rcp85])
            
            # Create a kernel density estimate
            kde_combined_data = stats.gaussian_kde(combined_data)
            
            # Define the grid for the contour plot
            xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
            ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
            xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
            zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
            
            # Specify the number of contour lines
            num_contours = 4
            
            # Generate evenly spaced contour levels based on the data
            contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
            # Plot the contours
            ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#993300')
           
            
            #Points showing fraction affected per extreme event
            #ax.scatter(timeseries_of_occurrence_of_extreme_event_1_rcp85, timeseries_of_occurrence_of_extreme_event_2_rcp85, s=3, c='darkred')
            
            # distribution          
            kde = stats.gaussian_kde(timeseries_of_occurrence_of_extreme_event_1_rcp85)
            xx = np.linspace(timeseries_of_occurrence_of_extreme_event_1_rcp85.min(), timeseries_of_occurrence_of_extreme_event_1_rcp85.max(), 1000)
            axu.plot(xx, kde(xx), color=(0.6, 0.204, 0.016))
            kde = stats. gaussian_kde(timeseries_of_occurrence_of_extreme_event_2_rcp85)
            yy = np.linspace(timeseries_of_occurrence_of_extreme_event_2_rcp85.min(), timeseries_of_occurrence_of_extreme_event_2_rcp85.max(), 1000)
            axr.plot(kde(yy), yy, color = (0.6, 0.204, 0.016))

            
        #if len(all_values_of_pearson_rcp85) 
        average_pearson_rcp85 = mean(all_values_of_pearson_rcp85)    
            
        #axl.legend(['early industrial: ' +'\u03C1'+ f'= {pearson_from_1861_until_1910:.3f}' ,'present-day: ' +'\u03C1'+ f'= {pearson_from_1956_until_2005:.3f}','rcp2.6: ' +'\u03C1'+ f'= {pearson_rcp26:.3f}', 'rcp6.0: ' +'\u03C1'+ f'= {pearson_rcp60:.3f}', 'rcp8.5: ' +'\u03C1'+ f'= {pearson_rcp85:.3f}'], fontsize='small')
        
        legend_elements = [Line2D([0], [0], marker ='o', color= (0.996, 0.89, 0.569), markerfacecolor=(0.996, 0.89, 0.569), label= 'early industrial: ' +'\u03C1'+ f'= {pearson_from_1861_until_1910:.3f}', markersize= 6) , Line2D([0], [0], marker ='o', color= (0.996, 0.769, 0.31), markerfacecolor=(0.996, 0.769, 0.31), label='present-day: ' +'\u03C1'+ f'= {pearson_from_1956_until_2005:.3f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.996, 0.6, 0.001), markerfacecolor=(0.996, 0.6, 0.001), label='rcp2.6: ' +'\u03C1'+ f'= {pearson_rcp26:.3f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.851, 0.373, 0.0549), markerfacecolor=(0.851, 0.373, 0.0549), label='rcp6.0: ' +'\u03C1'+ f'= {pearson_rcp60:.3f}', markersize= 6), Line2D([0], [0], marker ='o', color= (0.6, 0.204, 0.016), markerfacecolor=(0.6, 0.204, 0.016), label='rcp8.5: ' +'\u03C1'+ f'= {pearson_rcp85:.3f}', markersize =6)]
        
        axl.legend(handles = legend_elements, loc = 'center', fontsize = 9, frameon=False)
    
    # incase you want the title on the plot
    #axu.set_title('Pearson correlation coefficient considering fraction of region affected yearly by \n {} and {} \n '.format(event_1_name, event_2_name),fontsize=11)
    
    
    #plt.tight_layout()
    
    plt.gcf().text(0.12,0.03,'{}'.format(gcm), fontsize= 10)
    
    plt.tight_layout()
    
    # Change this directory to save the plots to your desired directory
    #plt.savefig('C:/Users/dmuheki/OneDrive/PhD/Masters_thesis_paper/Ongoing_results/Pearson correlation coefficient considering area of region affected yearly by {} and {} showing results from all impact models driven by {}.pdf'.format(event_1_name, event_2_name, gcm), dpi = 300, bbox_inches = 'tight')
    
    plt.show()
    plot_of_correlation = plt.show()   

    return plot_of_correlation         
                                                

#%% Function for plotting the correleation of two extreme events using the Pearson correlation coefficient
def plot_correlation_with_pearson_correlation_coefficient_considering_scatter_points_from_all_impact_models(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events, event_1_name, event_2_name, gcms):
    
    """

    Parameters
    ----------
    all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events : Tuple 
    event_1_name: String
    event_2_name: String
    gcms: List

    Returns
    -------
    Plot showing the correleation of two extreme events using the Pearson correlation coefficient.

    """
    
    
    for gcm in range(len(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events)):
        
        
        # Setting up 4 subplots as axis objects using Gridspec:
        gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios = [1,3])
        # Add space between scatter plot and KDE plots to accomodate axis labels
        gs.update(hspace =0.1, wspace =0.1)
        
        # Set background canvas color 
        fig = plt.figure(figsize = [8,6])
        fig.patch.set_facecolor('white')
        
        ax = plt.subplot(gs[1,0]) # Scatter plot area and axis range
        ax.set_xlabel('Percentage of area affected by {}'.format(event_1_name))
        ax.set_ylabel('Percentage of area affected by {}'.format(event_2_name))
        
        axu = plt.subplot(gs[0,0], sharex = ax) # Upper KDE plot area
        axu.get_xaxis().set_visible(False) # Hide the tick marks and spines
        axu.get_yaxis().set_visible(False)
        axu.spines['right'].set_visible(False)
        axu.spines['top'].set_visible(False)
        axu.spines['left'].set_visible(False)
        
        
        axr = plt.subplot(gs[1,1], sharey = ax) # Right KDE plot area
        axr.get_xaxis().set_visible(False) # Hide the tick marks and spines
        axr.get_yaxis().set_visible(False)
        axr.spines['right'].set_visible(False)
        axr.spines['top'].set_visible(False)
        axr.spines['bottom'].set_visible(False)
            
        axl = plt.subplot(gs[0,1]) # Legend plot area
        axl.axis('off') # Hide tick marks and spines
        
        
        
        # historical plot from 1861 until 1910 
        all_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = []
        all_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = []
        all_values_of_pearson_from_1861_until_1910 = []
        for i in range(len(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][0])):
                      
            timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][0][i][0]
            timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][0][i][1]
            
            if (np.var(timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)).values <= np.array([0.000001]):
                continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses. 
            
            all_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.append(timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.squeeze())
            all_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.append(timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.squeeze())
            
                   
            
        # Append full list of all timeseries into one 1D array representing the full scatter point data of all impact models driven by the same GCM. **Also known as pulling data    
        full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)
        full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)
        
        pearson_from_1861_until_1910 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910, full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910, ax, facecolor='none', edgecolor=(0.996, 0.89, 0.569))
        #all_values_of_pearson_from_1861_until_1910.append(pearson_from_1861_until_1910)
        
        # Stacking the two extreme event datasets 
        combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910, full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910])
        
        # Create a kernel density estimate
        kde_combined_data = stats.gaussian_kde(combined_data)
        
        # Define the grid for the contour plot
        xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
        ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
        xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
        zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Specify the number of contour lines
        num_contours = 7
        
        # Generate evenly spaced contour levels based on the data
        contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
        # Plot the contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#FEE491')
        
        
        # marginal distribution using KDE method    
        kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)
        xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.max(), 1000)
        axu.plot(xx, kde(xx), color=(0.996, 0.89, 0.569))
        kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)
        yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.max(), 1000)
        axr.plot(kde(yy), yy, color = (0.996, 0.89, 0.569))
        
        
        # Calculate the mean and variance of the individual extreme events and plot the values across the PDFs
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)
        fig.text(0.05, 2.2, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.996, 0.89, 0.569))
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)
        fig.text(1.5, 0.20, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.996, 0.89, 0.569)) 
         
  
        # historical plot from 1956 until 2005
        all_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = []
        all_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = []
        all_values_of_pearson_from_1956_until_2005 = []
        for i in range(len(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][1])):
                
            timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][1][i][0]
            timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][1][i][1]
            
            if  (np.var(timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)).values <= np.array([0.000001]):
                continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
            
            all_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.append(timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.squeeze())
            all_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.append(timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.squeeze())
            
        
        # Append full list of all timeseries into one 1D array representing the full scatter point data of all impact models driven by the same GCM. **Also known as pulling data    
        full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)
        full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)
        
        pearson_from_1956_until_2005 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005, full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005, ax, facecolor='none', edgecolor=(0.996, 0.769, 0.31))
        
        
        # Stacking the two extreme event datasets 
        combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005, full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005])
        
        # Create a kernel density estimate
        kde_combined_data = stats.gaussian_kde(combined_data)
        
        # Define the grid for the contour plot
        xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
        ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
        xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
        zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Specify the number of contour lines
        num_contours = 7
        
        # Generate evenly spaced contour levels based on the data
        contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
        # Plot the contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#FEC44F')
        
        
        # marginal distribution using KDE method 
        kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)
        xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.max(), 1000)
        axu.plot(xx, kde(xx), color=(0.996, 0.769, 0.31))
        kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)
        yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.max(), 1000)
        axr.plot(kde(yy), yy, color = (0.996, 0.769, 0.31))
             
        # Calculate the mean and variance of the individual extreme events and plot the values across the PDFs
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)
        fig.text(0.05, 2.0, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.996, 0.769, 0.31))
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)
        fig.text(1.5, 0.15, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.996, 0.769, 0.31)) 
        
        
        # projected scenarios....from 2050 until 2099
        # rcp26 plot
        all_timeseries_of_occurrence_of_extreme_event_1_rcp26 = []
        all_timeseries_of_occurrence_of_extreme_event_2_rcp26 = []
        
        all_values_of_pearson_rcp26 = []
        for i in range(len(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][2])):
        
            timeseries_of_occurrence_of_extreme_event_1_rcp26 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][2][i][0]
            timeseries_of_occurrence_of_extreme_event_2_rcp26 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][2][i][1]
            
            if  (np.var(timeseries_of_occurrence_of_extreme_event_1_rcp26)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_rcp26)).values <= np.array([0.000001]):
                continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
            
            all_timeseries_of_occurrence_of_extreme_event_1_rcp26.append(timeseries_of_occurrence_of_extreme_event_1_rcp26.squeeze())
            all_timeseries_of_occurrence_of_extreme_event_2_rcp26.append(timeseries_of_occurrence_of_extreme_event_2_rcp26.squeeze())
            

        
        # Append full list of all timeseries into one 1D array representing the full scatter point data of all impact models driven by the same GCM. **Also known as pulling data    
        full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_rcp26)
        full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_rcp26)
        
        pearson_rcp26 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26, ax, facecolor='none', edgecolor=(0.996, 0.6, 0.001))
        
        # Stacking the two extreme event datasets 
        combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26])
        
        # Create a kernel density estimate
        kde_combined_data = stats.gaussian_kde(combined_data)
        
        # Define the grid for the contour plot
        xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
        ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
        xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
        zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Specify the number of contour lines
        num_contours = 7
        
        # Generate evenly spaced contour levels based on the data
        contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
        # Plot the contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#FE9900')
        

        # marginal distribution using KDE method
        kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26)
        xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26.max(), 1000)
        axu.plot(xx, kde(xx), color=(0.996, 0.6, 0.001))
        kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26)
        yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26.max(), 1000)
        axr.plot(kde(yy), yy, color = (0.996, 0.6, 0.001))
        
        # Calculate the mean and variance of the individual extreme events and plot the values across the PDFs
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26)
        fig.text(0.05, 1.8, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.996, 0.6, 0.001))
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26)
        fig.text(1.5, 0.10, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.996, 0.6, 0.001)) 
            
        
        # rcp60 plot
        all_timeseries_of_occurrence_of_extreme_event_1_rcp60 = []
        all_timeseries_of_occurrence_of_extreme_event_2_rcp60 = []
        
        all_values_of_pearson_rcp60 = []
        for i in range(len(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][3])):
        
            timeseries_of_occurrence_of_extreme_event_1_rcp60 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][3][i][0]
            timeseries_of_occurrence_of_extreme_event_2_rcp60 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][3][i][1]
            
            if  (np.var(timeseries_of_occurrence_of_extreme_event_1_rcp60)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_rcp60)).values <= np.array([0.000001]):
                continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
            
            all_timeseries_of_occurrence_of_extreme_event_1_rcp60.append(timeseries_of_occurrence_of_extreme_event_1_rcp60.squeeze())
            all_timeseries_of_occurrence_of_extreme_event_2_rcp60.append(timeseries_of_occurrence_of_extreme_event_2_rcp60.squeeze())
            

        full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_rcp60)
        full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_rcp60)
        
        pearson_rcp60 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60, ax, facecolor='none', edgecolor=(0.851, 0.373, 0.0549))
        
        # Stacking the two extreme event datasets 
        combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60])
        
        # Create a kernel density estimate
        kde_combined_data = stats.gaussian_kde(combined_data)
        
        # Define the grid for the contour plot
        xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
        ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
        xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
        zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Specify the number of contour lines
        num_contours = 7
        
        # Generate evenly spaced contour levels based on the data
        contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
        # Plot the contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#D95F0E')
       
        
        # marginal distribution using KDE method
        kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60)
        xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60.max(), 1000)
        axu.plot(xx, kde(xx), color=(0.851, 0.373, 0.0549))
        kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60)
        yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60.max(), 1000)
        axr.plot(kde(yy), yy, color = (0.851, 0.373, 0.0549))
        
        # Calculate the mean and variance of the individual extreme events and plot the values across the PDFs
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60)
        fig.text(0.05, 1.6, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.851, 0.373, 0.0549))
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60)
        fig.text(1.5, 0.05, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.851, 0.373, 0.0549)) 
        
        ax.set_xlim(-1, full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60.max()+9)
        ax.set_ylim(-1, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60.max()+9)
        #ax.set_aspect('auto')
        
        legend_elements = [Line2D([0], [0], marker ='o', color= (0.996, 0.89, 0.569), markerfacecolor=(0.996, 0.89, 0.569), label= 'early industrial: ' +'\u03C1'+ f'= {pearson_from_1861_until_1910:.3f}', markersize= 6) , Line2D([0], [0], marker ='o', color= (0.996, 0.769, 0.31), markerfacecolor=(0.996, 0.769, 0.31), label='present-day: ' +'\u03C1'+ f'= {pearson_from_1956_until_2005:.3f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.996, 0.6, 0.001), markerfacecolor=(0.996, 0.6, 0.001), label='rcp2.6: ' +'\u03C1'+ f'= {pearson_rcp26:.3f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.851, 0.373, 0.0549), markerfacecolor=(0.851, 0.373, 0.0549), label='rcp6.0: ' +'\u03C1'+ f'= {pearson_rcp60:.3f}', markersize= 6)]
        
        axl.legend(handles = legend_elements, loc = 'center', fontsize = 8, frameon=False)
        
        #if len(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events) == 5:
        if len(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][4]) != 0:
            
            # rcp85 plot
            all_timeseries_of_occurrence_of_extreme_event_1_rcp85 = []
            all_timeseries_of_occurrence_of_extreme_event_2_rcp85 = []
            
            all_values_of_pearson_rcp85 = []
            for i in range(len(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][4])):
            
                timeseries_of_occurrence_of_extreme_event_1_rcp85 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][4][i][0]
                timeseries_of_occurrence_of_extreme_event_2_rcp85 = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm][4][i][1]
                
                if  (np.var(timeseries_of_occurrence_of_extreme_event_1_rcp85)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_rcp85)).values <= np.array([0.000001]):
                    continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
                
                all_timeseries_of_occurrence_of_extreme_event_1_rcp85.append(timeseries_of_occurrence_of_extreme_event_1_rcp85.squeeze())
                all_timeseries_of_occurrence_of_extreme_event_2_rcp85.append(timeseries_of_occurrence_of_extreme_event_2_rcp85.squeeze())
                
            
            full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_rcp85)
            full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_rcp85)
            
            pearson_rcp85 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85, ax, facecolor='none', edgecolor=(0.6, 0.204, 0.016))
           
            # Stacking the two extreme event datasets 
            combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85])
            
            # Create a kernel density estimate
            kde_combined_data = stats.gaussian_kde(combined_data)
            
            # Define the grid for the contour plot
            xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
            ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
            xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
            zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
            
            # Specify the number of contour lines
            num_contours = 7
            
            # Generate evenly spaced contour levels based on the data
            contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
            # Plot the contours
            ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#993300')
           
           
            # marginal distribution using KDE method
            kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85)
            xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85.max(), 1000)
            axu.plot(xx, kde(xx), color=(0.6, 0.204, 0.016))
            kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85)
            yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85.max(), 1000)
            axr.plot(kde(yy), yy, color = (0.6, 0.204, 0.016))
                
            # Calculate the mean and variance of the individual extreme events 
            mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85)
            variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85)
            fig.text(0.05, 1.4, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.6, 0.204, 0.016))
            mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85)
            variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85)
            fig.text(1.5, 0, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.6, 0.204, 0.016))

            ax.set_xlim(-1, full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85.max()+9)
            ax.set_ylim(-1, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85.max()+9)
            #ax.set_aspect('auto')
            
            legend_elements = [Line2D([0], [0], marker ='o', color= (0.996, 0.89, 0.569), markerfacecolor=(0.996, 0.89, 0.569), label= 'early industrial: ' +'\u03C1'+ f'= {pearson_from_1861_until_1910:.3f}', markersize= 6) , Line2D([0], [0], marker ='o', color= (0.996, 0.769, 0.31), markerfacecolor=(0.996, 0.769, 0.31), label='present-day: ' +'\u03C1'+ f'= {pearson_from_1956_until_2005:.3f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.996, 0.6, 0.001), markerfacecolor=(0.996, 0.6, 0.001), label='rcp2.6: ' +'\u03C1'+ f'= {pearson_rcp26:.3f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.851, 0.373, 0.0549), markerfacecolor=(0.851, 0.373, 0.0549), label='rcp6.0: ' +'\u03C1'+ f'= {pearson_rcp60:.3f}', markersize= 6), Line2D([0], [0], marker ='o', color= (0.6, 0.204, 0.016), markerfacecolor=(0.6, 0.204, 0.016), label='rcp8.5: ' +'\u03C1'+ f'= {pearson_rcp85:.3f}', markersize =6)]
            
            axl.legend(handles = legend_elements, loc = 'center', fontsize = 9, frameon=False)
        
        
        # incase you want the title on the plot
        #axu.set_title('Pearson correlation coefficient considering fraction of region affected yearly by \n {} and {} \n '.format(event_1_name, event_2_name),fontsize=11)
        

        #plt.tight_layout()
        
        # driving GCM noted on the plot
        plt.gcf().text(0.12,0.03,'considering all impact models driven by {}'.format(gcms[gcm]), fontsize= 10)
        
        plt.tight_layout()
        
        # Change this directory to save the plots to your desired directory
        #plt.savefig('C:/Users/dmuheki/OneDrive/PhD/Masters_thesis_paper/Ongoing_results/Pearson correlation coefficient considering areea of region affected yearly by {} and {} considering impact models driven by {}.pdf'.format(event_1_name, event_2_name, gcms[gcm]), dpi = 300, bbox_inches = 'tight')
        
        plt.show()
        plot_of_correlation = plt.show()   
    
    
    return plot_of_correlation         
                                
    
#%% Function for plotting the correleation of two extreme events using the Pearson correlation coefficient using all the GCM data 
def plot_correlation_with_pearson_correlation_coefficient_considering_scatter_points_from_all_impact_models_and_all_gcms(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events, event_1_name, event_2_name):
    
    """
    Parameters
    -------
    all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events: Tuple 
    event_1_name: String
    event_2_name: String

    Returns
    -------
    Plot showing the correleation of two extreme events using the Pearson correlation coefficient considering all impact models and all their driving GCMs.

    """
     
    
    all_data_considering_all_gcms = [[],[],[],[],[]] #To enable sorting all the data from all the GCMS into their respective scenarios: i.e early industrial, present day, rcp 2.6, ...etc
    for gcm in range(len(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events)):
        gcm_data = all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events[gcm]
        for scenario in range(len(gcm_data)):
            data_per_gcm = gcm_data[scenario]
            all_data_considering_all_gcms[scenario].extend(data_per_gcm)
    
    # Setting up 4 subplots as axis objects using Gridspec:
    gs = gridspec.GridSpec(2, 2, width_ratios=[3,1], height_ratios = [1,3])
    # Add space between scatter plot and KDE plots to accomodate axis labels
    gs.update(hspace =0.1, wspace =0.1)
    
    # Set background canvas color 
    fig = plt.figure(figsize = [8,6])
    fig.patch.set_facecolor('white')
    
    ax = plt.subplot(gs[1,0]) # Scatter plot area and axis range
    ax.set_xlabel('Percentage of area affected by {}'.format(event_1_name))
    ax.set_ylabel('Percentage of area affected by {}'.format(event_2_name))
    
    axu = plt.subplot(gs[0,0], sharex = ax) # Upper KDE plot area
    axu.get_xaxis().set_visible(False) # Hide the tick marks and spines
    axu.get_yaxis().set_visible(False)
    axu.spines['right'].set_visible(False)
    axu.spines['top'].set_visible(False)
    axu.spines['left'].set_visible(False)
    
    
    axr = plt.subplot(gs[1,1], sharey = ax) # Right KDE plot area
    axr.get_xaxis().set_visible(False) # Hide the tick marks and spines
    axr.get_yaxis().set_visible(False)
    axr.spines['right'].set_visible(False)
    axr.spines['top'].set_visible(False)
    axr.spines['bottom'].set_visible(False)
        
    axl = plt.subplot(gs[0,1]) # Legend plot area
    axl.axis('off') # Hide tick marks and spines
    
    
    
    # historical plot from 1861 until 1910 
    all_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = []
    all_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = []
      
    for compound_event in range(len(all_data_considering_all_gcms[0])): 
        timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = all_data_considering_all_gcms[0][compound_event][0]
        timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = all_data_considering_all_gcms[0][compound_event][1]
        
        if (np.var(timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)).values <= np.array([0.000001]):
            continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses. 
        
        all_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.append(timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.squeeze())
        all_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.append(timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.squeeze())
        
        
    # Append full list of all timeseries into one 1D array representing the full scatter point data of all impact models driven by the same GCM. **Also known as pulling data    
    full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)
    full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)
    
    pearson_from_1861_until_1910 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910, full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910, ax, facecolor='none', edgecolor=(0.996, 0.89, 0.569))

    
    # Stacking the two extreme event datasets 
    combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910, full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910])
    
    # Create a kernel density estimate
    kde_combined_data = stats.gaussian_kde(combined_data)
    
    # Define the grid for the contour plot
    xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
    ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
    xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
    zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
    
    # Specify the number of contour lines
    num_contours = 10
    
    # Generate evenly spaced contour levels based on the data
    contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
    # Plot the contours
    ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#FEE491')

    
    # marginal distribution using KDE method    
    kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)
    xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910.max(), 1000)
    axu.plot(xx, kde(xx), color=(0.996, 0.89, 0.569))
    kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)
    yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910.max(), 1000)
    axr.plot(kde(yy), yy, color = (0.996, 0.89, 0.569))
    
    # Calculate the mean and variance of the individual extreme events and plot the values across the PDFs
    mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)
    variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910)
    fig.text(0.05, 2.2, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1861_until_1910:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.996, 0.89, 0.569))
    mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)
    variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910)
    fig.text(1.5, 0.20, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1861_until_1910:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.996, 0.89, 0.569)) 
      
    

    # historical plot from 1956 until 2005
    all_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = []
    all_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = []

    for compound_event in range(len(all_data_considering_all_gcms[1])): 
        timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = all_data_considering_all_gcms[1][compound_event][0]
        timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = all_data_considering_all_gcms[1][compound_event][1]
    
        if  (np.var(timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)).values <= np.array([0.000001]):
            continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
        
        all_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.append(timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.squeeze())
        all_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.append(timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.squeeze())
        

    # Append full list of all timeseries into one 1D array representing the full scatter point data of all impact models driven by the same GCM. **Also known as pulling data    
    full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)
    full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)
    
    # To calculate pearsons correlation coefficient 
    pearson_from_1956_until_2005 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005, full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005, ax, facecolor='none', edgecolor=(0.996, 0.769, 0.31))
    
    
    # Stacking the two extreme event datasets 
    combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005, full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005])
    
    # Create a kernel density estimate
    kde_combined_data = stats.gaussian_kde(combined_data)
    
    # Define the grid for the contour plot
    xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
    ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
    xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
    zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
    
    # Specify the number of contour lines
    num_contours = 10
    
    # Generate evenly spaced contour levels based on the data
    contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
    # Plot the contours
    ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#FEC44F')
    
        
    # marginal distribution using KDE method 
    kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)
    xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005.max(), 1000)
    axu.plot(xx, kde(xx), color=(0.996, 0.769, 0.31))
    kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)
    yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005.max(), 1000)
    axr.plot(kde(yy), yy, color = (0.996, 0.769, 0.31))
 
    # Calculate the mean and variance of the individual extreme events and plot the values across the PDFs
    mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)
    variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005)
    fig.text(0.05, 2.0, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_from_1956_until_2005:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.996, 0.769, 0.31))
    mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)
    variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005)
    fig.text(1.5, 0.15, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_from_1956_until_2005:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.996, 0.769, 0.31)) 
    
    
    # projected scenarios....from 2050 until 2099
    # rcp26 plot
    all_timeseries_of_occurrence_of_extreme_event_1_rcp26 = []
    all_timeseries_of_occurrence_of_extreme_event_2_rcp26 = []

    for compound_event in range(len(all_data_considering_all_gcms[2])): 
        timeseries_of_occurrence_of_extreme_event_1_rcp26 = all_data_considering_all_gcms[2][compound_event][0]
        timeseries_of_occurrence_of_extreme_event_2_rcp26 = all_data_considering_all_gcms[2][compound_event][1]
        
        if  (np.var(timeseries_of_occurrence_of_extreme_event_1_rcp26)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_rcp26)).values <= np.array([0.000001]):
            continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
        
        all_timeseries_of_occurrence_of_extreme_event_1_rcp26.append(timeseries_of_occurrence_of_extreme_event_1_rcp26.squeeze())
        all_timeseries_of_occurrence_of_extreme_event_2_rcp26.append(timeseries_of_occurrence_of_extreme_event_2_rcp26.squeeze())
        
    
    # Append full list of all timeseries into one 1D array representing the full scatter point data of all impact models driven by the same GCM. **Also known as pulling data    
    full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_rcp26)
    full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_rcp26)
    
    # To calculate pearsons correlation coefficient
    pearson_rcp26 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26, ax, facecolor='none', edgecolor=(0.996, 0.6, 0.001))

    # Stacking the two extreme event datasets 
    combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26])
    
    # Create a kernel density estimate
    kde_combined_data = stats.gaussian_kde(combined_data)
    
    # Define the grid for the contour plot
    xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
    ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
    xi, yi = np.mgrid[xmin-5:xmax+10:200j, ymin-5:ymax+10:200j]
    zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
    
    # Specify the number of contour lines
    num_contours = 10
    
    # Generate evenly spaced contour levels based on the data
    contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
    # Plot the contours
    ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#FE9900')
    
    
    # marginal distribution using KDE method
    kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26)
    xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26.max(), 1000)
    axu.plot(xx, kde(xx), color=(0.996, 0.6, 0.001))
    kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26)
    yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26.max(), 1000)
    axr.plot(kde(yy), yy, color = (0.996, 0.6, 0.001))
    
    #average_pearson_rcp26 = mean(all_values_of_pearson_rcp26) # Mean Pearson's correlataion coefficient
        
    # Calculate the mean and variance of the individual extreme events and plot the values across the PDFs
    mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26)
    variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26)
    fig.text(0.05, 1.8, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp26:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.996, 0.6, 0.001))
    mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26)
    variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26)
    fig.text(1.5, 0.10, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp26:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.996, 0.6, 0.001)) 
    
    
    # rcp60 plot
    all_timeseries_of_occurrence_of_extreme_event_1_rcp60 = []
    all_timeseries_of_occurrence_of_extreme_event_2_rcp60 = [] 

    for compound_event in range(len(all_data_considering_all_gcms[3])): 
        timeseries_of_occurrence_of_extreme_event_1_rcp60 = all_data_considering_all_gcms[3][compound_event][0]
        timeseries_of_occurrence_of_extreme_event_2_rcp60 = all_data_considering_all_gcms[3][compound_event][1]
    
        if  (np.var(timeseries_of_occurrence_of_extreme_event_1_rcp60)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_rcp60)).values <= np.array([0.000001]):
            continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
        
        all_timeseries_of_occurrence_of_extreme_event_1_rcp60.append(timeseries_of_occurrence_of_extreme_event_1_rcp60.squeeze())
        all_timeseries_of_occurrence_of_extreme_event_2_rcp60.append(timeseries_of_occurrence_of_extreme_event_2_rcp60.squeeze())
        
        
    full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_rcp60)
    full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_rcp60)
    
    # To calculate pearsons corelation coefficient
    pearson_rcp60 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60, ax, facecolor='none', edgecolor=(0.85, 0.37, 0.05))
   
    
    # Stacking the two extreme event datasets 
    combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60])
    
    # Create a kernel density estimate
    kde_combined_data = stats.gaussian_kde(combined_data)
    
    # Define the grid for the contour plot
    xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
    ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
    xi, yi = np.mgrid[xmin-4:xmax+10:200j, ymin-5:ymax+10:200j]
    zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
    
    # Specify the number of contour lines
    num_contours = 10
    
    # Generate evenly spaced contour levels based on the data
    contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
    # Plot the contours
    ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#D95F0E')
   
    
   
    # marginal distribution using KDE method
    kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60)
    xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60.max(), 1000)
    axu.plot(xx, kde(xx), color=(0.851, 0.373, 0.0549))
    kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60)
    yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60.max(), 1000)
    axr.plot(kde(yy), yy, color = (0.851, 0.373, 0.0549))
        

    # Calculate the mean and variance of the individual extreme events and plot the values across the PDFs
    mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60)
    variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60)
    fig.text(0.05, 1.6, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.851, 0.373, 0.0549))
    mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60)
    variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60)
    fig.text(1.5, 0.05, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.851, 0.373, 0.0549)) 
    
    ax.set_xlim(-2, full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp60.max()+9)
    ax.set_ylim(-2, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp60.max()+9)
    #ax.set_aspect('auto')
    
    legend_elements = [Line2D([0], [0], marker ='o', color= (0.996, 0.89, 0.569), markerfacecolor=(0.996, 0.89, 0.569), label= 'early industrial: ' +'\u03C1'+ f'= {pearson_from_1861_until_1910:.2f}', markersize= 6) , Line2D([0], [0], marker ='o', color= (0.996, 0.769, 0.31), markerfacecolor=(0.996, 0.769, 0.31), label='present-day: ' +'\u03C1'+ f'= {pearson_from_1956_until_2005:.2f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.996, 0.6, 0.001), markerfacecolor=(0.996, 0.6, 0.001), label='rcp2.6: ' +'\u03C1'+ f'= {pearson_rcp26:.2f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.851, 0.373, 0.0549), markerfacecolor=(0.851, 0.373, 0.0549), label='rcp6.0: ' +'\u03C1'+ f'= {pearson_rcp60:.2f}', markersize= 6)]
    
    axl.legend(handles = legend_elements, loc = 'center', fontsize = 10, frameon=False)
    
    #if len(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events) == 5:
    if len(all_data_considering_all_gcms[4]) != 0: # checking for an empty list, incase of crop failure data that doesnt have the rcp 8.5 scenario
        
        # rcp85 plot
        all_timeseries_of_occurrence_of_extreme_event_1_rcp85 = []
        all_timeseries_of_occurrence_of_extreme_event_2_rcp85 = []
        
        for compound_event in range(len(all_data_considering_all_gcms[4])): 
            timeseries_of_occurrence_of_extreme_event_1_rcp85 = all_data_considering_all_gcms[4][compound_event][0]
            timeseries_of_occurrence_of_extreme_event_2_rcp85 = all_data_considering_all_gcms[4][compound_event][1]
        
            if  (np.var(timeseries_of_occurrence_of_extreme_event_1_rcp85)).values <= np.array([0.000001]) or (np.var(timeseries_of_occurrence_of_extreme_event_2_rcp85)).values <= np.array([0.000001]):
                continue  # avoid datasets without variation in time in the bivariate distribution plot and plotting of the ellipses
            
            all_timeseries_of_occurrence_of_extreme_event_1_rcp85.append(timeseries_of_occurrence_of_extreme_event_1_rcp85.squeeze())
            all_timeseries_of_occurrence_of_extreme_event_2_rcp85.append(timeseries_of_occurrence_of_extreme_event_2_rcp85.squeeze())
            
        
        full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_1_rcp85)
        full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85 = np.concatenate(all_timeseries_of_occurrence_of_extreme_event_2_rcp85)
        
        # To calculate the pearsons correlation coefficient
        pearson_rcp85 = confidence_ellipse(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85, ax, facecolor='none', edgecolor=(0.60, 0.20, 0.016))
        
        # Stacking the two extreme event datasets 
        combined_data= np.vstack([full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85])
        
        # Create a kernel density estimate
        kde_combined_data = stats.gaussian_kde(combined_data)
        
        # Define the grid for the contour plot
        xmin, xmax = np.min(combined_data[0]), np.max(combined_data[0])
        ymin, ymax = np.min(combined_data[1]), np.max(combined_data[1])
        xi, yi = np.mgrid[xmin-4:xmax+10:200j, ymin-5:ymax+10:200j]
        zi = kde_combined_data(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Specify the number of contour lines
        num_contours = 10
        
        # Generate evenly spaced contour levels based on the data
        contour_levels = np.linspace(np.min(zi), np.max(zi), num=num_contours)
        # Plot the contours
        ax.contour(xi, yi, zi.reshape(xi.shape), levels = contour_levels[1:2], linestyles = 'dashed', colors='#993300')
        
       
        # marginal distribution using KDE method 
        kde = stats.gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85)
        xx = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85.max(), 1000)
        axu.plot(xx, kde(xx), color=(0.6, 0.204, 0.016))
        kde = stats. gaussian_kde(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85)
        yy = np.linspace(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85.min(), full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85.max(), 1000)
        axr.plot(kde(yy), yy, color = (0.6, 0.204, 0.016))
        
        # Calculate the mean and variance of the individual extreme events 
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85)
        fig.text(0.05, 1.4, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85:.2f}", ha='left', va='top', transform=axu.transAxes, color= (0.6, 0.204, 0.016))
        mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85 = np.mean(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85)
        variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85 = np.var(full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85)
        fig.text(1.5, 0, f"x\u0304: {mean_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85:.2f}, \u03C3\u00B2: {variance_full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85:.2f}", ha='left', va='bottom', transform=axr.transAxes, color= (0.6, 0.204, 0.016))

        ax.set_xlim(-1, full_scatter_timeseries_of_occurrence_of_extreme_event_1_rcp85.max()+9)
        ax.set_ylim(-1, full_scatter_timeseries_of_occurrence_of_extreme_event_2_rcp85.max()+9)
        #ax.set_aspect('auto')
                
        legend_elements = [Line2D([0], [0], marker ='o', color= (0.996, 0.89, 0.569), markerfacecolor=(0.996, 0.89, 0.569), label= 'early industrial: ' +'\u03C1'+ f'= {pearson_from_1861_until_1910:.2f}', markersize= 6) , Line2D([0], [0], marker ='o', color= (0.996, 0.769, 0.31), markerfacecolor=(0.996, 0.769, 0.31), label='present-day: ' +'\u03C1'+ f'= {pearson_from_1956_until_2005:.2f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.996, 0.6, 0.001), markerfacecolor=(0.996, 0.6, 0.001), label='rcp2.6: ' +'\u03C1'+ f'= {pearson_rcp26:.2f}', markersize = 6), Line2D([0], [0], marker ='o', color= (0.851, 0.373, 0.0549), markerfacecolor=(0.851, 0.373, 0.0549), label='rcp6.0: ' +'\u03C1'+ f'= {pearson_rcp60:.2f}', markersize= 6), Line2D([0], [0], marker ='o', color= (0.6, 0.204, 0.016), markerfacecolor=(0.6, 0.204, 0.016), label='rcp8.5: ' +'\u03C1'+ f'= {pearson_rcp85:.2f}', markersize =6)]
        
        
        axl.legend(handles = legend_elements, loc = 'center', fontsize = 10, frameon=False)
    
    # incase you require a title on the plot
    #axu.set_title('Pearson correlation coefficient considering fraction of region affected yearly by \n {} and {} \n '.format(event_1_name, event_2_name),fontsize=11)
    
    plt.tight_layout()
    
    # Change this directory to save the plots to your desired directory
    #plt.savefig('C:/Users/dmuheki/OneDrive/PhD/Masters_thesis_paper/Ongoing_results/Pearson correlation coefficient considering fraction of region affected yearly by {} and {} considering all impact models and their driving GCMs.pdf'.format(event_1_name, event_2_name), dpi = 300, bbox_inches = 'tight')
    
    plt.show()
    plot_of_correlation = plt.show()   

    return plot_of_correlation






#%% Function for calculating the Probability of occurrence of an extreme event in one grid cell over the entire dataset period
def probability_of_occurrence_of_extreme_event(extreme_event_occurrence):
    
    """ Determine the probability of occurrence of an extreme climate event over the entire dataset time period
    
    Parameters
    ----------
    extreme_event_occurrence : Xarray data array (boolean with true for locations with the occurrence of an extreme event within the same year)
    
    Returns
    -------
    Probability of occurrence of an extreme climate event over the entire dataset time period (Xarray)
    
    """
    # Number of years with extreme event occurrence  
    no_of_years_with_extreme_event_occurrence = (extreme_event_occurrence).sum('time',skipna=False)
    
    # Total number of years in the dataset
    total_no_of_years_in_dataset = len(extreme_event_occurrence)
    
    # Probability of occurrence
    probability_of_occurrence_of_the_extreme_event = no_of_years_with_extreme_event_occurrence/total_no_of_years_in_dataset
    
    
    return probability_of_occurrence_of_the_extreme_event


#%% Function for plotting map showing the Probability Ratio (PR) of occurrence of an extreme event in one grid cell over the entire dataset period
def plot_probability_ratio_of_occurrence_of_an_extreme_event(probability_of_occurrence_of_extreme_event_in_new_scenario, probability_of_occurrence_of_same_extreme_event_in_a_past_scenario, event_name, time_period, gcm, scenario):
    
    """ Plot a map showing the probability of occurrence of an extreme climate event over the entire dataset time period
    
    Parameters
    ----------
    probability_of_occurrence_of_extreme_event_in_new_scenario : Xarray data array (probability of occurrence of an extreme event over the entire dataset time period in new scenario)
    probability_of_occurrence_of_same_extreme_event_in_a_past_scenario : Xarray data array (probability of occurrence of the same extreme event over the entire dataset time period in historical early industrial time period / reference dataset)
    event_name : String (Extreme Events)
    gcm : String (Driving GCM)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the Probability Ratio (PR) of joint occurrence of two extreme climate events to compare the change in new scenario from the historical early industrial times
    
    """
    
    
    # Probability Ratio (PR) of occurrence: ratio between probability of occurrence of an extreme event in new situation (scenario) to the probability of occurrence in the reference situtaion (historical early industrial times)
    probability_of_ratio_of_occurrence_of_an_extreme_event = probability_of_occurrence_of_extreme_event_in_new_scenario/probability_of_occurrence_of_same_extreme_event_in_a_past_scenario
    
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = True, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    boundaries = [1, 2, 4, 6, 8, 10]
    n_colors = 6
    cmap = sns.color_palette('YlOrRd', n_colors=n_colors) # set color palette

    cmap = mcolors.ListedColormap(['cornflowerblue']+ [cmap[0]] + [cmap[1]]+ [cmap[2]]+ [cmap[3]] + [cmap[4]]+ ['darkred']) # manually select colors from palette
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=len(boundaries)+1, extend='both')
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
  
    formatter = ticker.FuncFormatter(lambda x, pos: str(Fraction(x).limit_denominator()) if x <= 1 else '{:.0f}'.format(x)) # convert the floats to fractions for those less than 1 and to integers for those above one in the color bar legend

    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(probability_of_ratio_of_occurrence_of_an_extreme_event, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = cmap, norm= norm)
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.265,0.9,'Comparing {} ({}) to 1861-1910'.format(time_period, scenario), fontsize = 9)
    plt.gcf().text(0.25,0.03,'{}'.format(gcm), fontsize= 10)
    
    # Add the title and legend to the figure and show the figure
    plt.title('Probability Ratio of Occurrence of {} \n asssuming change in joint occurrence due to changes only in these events \n'.format(event_name),fontsize=10) #Plot title
    
    # discrete color bar legend
    #bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colorbar = plt.colorbar(cbar, orientation = 'vertical', extend='both', extendfrac= 'auto', ticks=boundaries, format=formatter)
    colorbar.set_label(label = 'Probability Ratio', size = 11) #Plots the legend color bar
    
    plt.clim(0,10)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    plt.show()
    #plt.close()
    
    return probability_of_ratio_of_occurrence_of_an_extreme_event


#%% Function for plotting map showing the Probability Ratio (PR) of occurrence of an extreme event in one grid cell over the entire dataset period
def plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(probability_of_occurrence_of_extreme_event_in_new_scenario, probability_of_occurrence_of_same_extreme_event_in_a_past_scenario, event_name, second_event_name, time_period, scenario):
    
    """ Plot a map showing the probability of occurrence of an extreme climate event over the entire dataset time period
    
    Parameters
    ----------
    probability_of_occurrence_of_extreme_event_in_new_scenario : Xarray data array (probability of occurrence of an extreme event over the entire dataset time period in new scenario)
    probability_of_occurrence_of_same_extreme_event_in_a_past_scenario : Xarray data array (probability of occurrence of the same extreme event over the entire dataset time period in historical early industrial time period / reference dataset)
    event_name and second_event_name : String (Extreme Events)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the Probability Ratio (PR) of joint occurrence of two extreme climate events to compare the change in new scenario from the historical early industrial times
    
    """
    
    
    # Probability Ratio (PR) of occurrence: ratio between probability of occurrence of an extreme event in new situation (scenario) to the probability of occurrence in the reference situtaion (historical early industrial times)
    probability_of_ratio_of_occurrence_of_an_extreme_event = probability_of_occurrence_of_extreme_event_in_new_scenario/probability_of_occurrence_of_same_extreme_event_in_a_past_scenario
    
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = True, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Set the legend limits  
    boundaries = [1, 2, 4, 6, 8, 10]
    n_colors = 6
    cmap = sns.color_palette('YlOrRd', n_colors=n_colors) # set color palette

    cmap = mcolors.ListedColormap(['cornflowerblue']+ [cmap[0]] + [cmap[1]]+ [cmap[2]]+ [cmap[3]] + [cmap[4]]+ ['darkred']) # manually select colors from palette
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=len(boundaries)+1, extend='both')
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
  
    formatter = ticker.FuncFormatter(lambda x, pos: str(Fraction(x).limit_denominator()) if x <= 1 else '{:.0f}'.format(x)) # convert the floats to fractions for those less than 1 and to integers for those above one in the color bar legend

    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(probability_of_ratio_of_occurrence_of_an_extreme_event, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = cmap, norm= norm)
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.265,0.9,'Comparing {} ({}) to 1861-1910'.format(time_period, scenario), fontsize = 9)
    
    
    # Add the title and legend to the figure and show the figure
    #plt.title('Probability Ratio of Occurrence of {} \n asssuming change in joint occurrence due to changes only in these events \n'.format(event_name),fontsize=10) #Plot title
    
    
    colorbar = plt.colorbar(cbar, orientation = 'vertical', extend='both', extendfrac= 'auto', ticks=boundaries, format=formatter)
    colorbar.set_label(label = 'Probability Ratio', size = 11) #Plots the legend color bar
    
    plt.clim(0,10)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    
    # Change this directory to save the plots to your desired directory
    #plt.savefig('C:/Users/dmuheki/OneDrive/PhD/Masters_thesis_paper/Ongoing_results/PR of Occurrence of {} asssuming the change in joint occurrence with {} is due to changes only in {} considering {} as the future scenario.pdf'.format(event_name, second_event_name, event_name, scenario), dpi = 300)
    
    plt.show()
    #plt.close()
    
    return probability_of_ratio_of_occurrence_of_an_extreme_event




#%% Function for plotting map showing the Probability Ratio (PR) of occurrence of an extreme event in one grid cell over the entire dataset period
def plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only(probability_of_occurrence_of_extreme_event_1_in_new_scenario, probability_of_occurrence_of_extreme_event_2_in_new_scenario, probability_of_occurrence_of_extreme_event_1_in_a_past_scenario, probability_of_occurrence_of_extreme_event_2_in_a_past_scenario, probability_ratio_of_occurrence_of_the_compound_events_in_new_scenario, probability_ratio_of_occurrence_of_the_compound_events_in_a_past_scenario, event_1_name, event_2_name, time_period, gcm, scenario):
    
    """ Plot a map showing the probability of occurrence of a compound extreme climate event over the entire dataset time period
    
    Parameters
    ----------
    probability_of_occurrence_of_extreme_event_1_in_new_scenario : Xarray data array (probability of occurrence of extreme event 1 over the entire dataset time period in new scenario)
    probability_of_occurrence_of_extreme_event_2_in_new_scenario : Xarray data array (probability of occurrence of extreme event 2 over the entire dataset time period in new scenario)
    probability_of_occurrence_of_extreme_event_1_in_a_past_scenario : Xarray data array (probability_of_occurrence_of_extreme_event_1_in_a_past_scenario over the entire dataset time period in historical early industrial time period)
    probability_of_occurrence_of_extreme_event_2_in_a_past_scenario : Xarray data array (probability_of_occurrence_of_extreme_event_2_in_a_past_scenario over the entire dataset time period in historical early industrial time period)
    probability_ratio_of_occurrence_of_the_compound_events_in_new_scenario : Xarray data array
    probability_ratio_of_occurrence_of_the_compound_events_in_a_past_scenario : Xarray data array
    
    event_1_name, event_2_name : String (Extreme Events)
    gcm : String (Driving GCM)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the Probability Ratio (PR) of joint occurrence of two extreme climate events assuming DEPENDENCE ONLY to compare the change in new scenario from the historical early industrial times
    
    """
    
    
    # Probability Ratio (PR) of occurrence: ratio between probability of occurrence of compound event in new situation (scenario) to the probability of occurrence in the reference situtaion (historical early industrial times)
    probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = ((probability_ratio_of_occurrence_of_the_compound_events_in_new_scenario/(probability_of_occurrence_of_extreme_event_1_in_new_scenario * probability_of_occurrence_of_extreme_event_2_in_new_scenario))/(probability_ratio_of_occurrence_of_the_compound_events_in_a_past_scenario/(probability_of_occurrence_of_extreme_event_1_in_a_past_scenario * probability_of_occurrence_of_extreme_event_2_in_a_past_scenario)))
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = True, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Set the legend limits  
    boundaries = [1, 2, 4, 6, 8, 10]
    n_colors = 6
    cmap = sns.color_palette('YlOrRd', n_colors=n_colors) # set color palette

    cmap = mcolors.ListedColormap(['cornflowerblue']+ [cmap[0]] + [cmap[1]]+ [cmap[2]]+ [cmap[3]] + [cmap[4]]+ ['darkred']) # manually select colors from palette
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=len(boundaries)+1, extend='both')
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
  
    formatter = ticker.FuncFormatter(lambda x, pos: str(Fraction(x).limit_denominator()) if x <= 1 else '{:.0f}'.format(x)) # convert the floats to fractions for those less than 1 and to integers for those above one in the color bar legend

    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = cmap, norm= norm)
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.265,0.9,'Comparing {} ({}) to 1861-1910'.format(time_period, scenario), fontsize = 9)
    plt.gcf().text(0.25,0.03,'{}'.format(gcm), fontsize= 10)
    
    # Add the title and legend to the figure and show the figure
    plt.title('Probability Ratio of Joint Occurrence of {} and {} \n assuming only change in dependence \n'.format(event_1_name, event_2_name),fontsize=10) #Plot title
    
    # discrete color bar legend
    #bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    colorbar = plt.colorbar(cbar, orientation = 'vertical', extend='both', extendfrac= 'auto', ticks=boundaries, format=formatter)#Plots the legend color bar
    colorbar.set_label(label = 'Probability Ratio', size = 11)
    
    plt.clim(0,10)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    plt.show()
    #plt.close()
    
    return probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only


#%% Function for plotting map showing the Probability Ratio (PR) of occurrence of an extreme event in one grid cell over the entire dataset period
def plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only_considering_all_gcms(probability_of_occurrence_of_extreme_event_1_in_new_scenario, probability_of_occurrence_of_extreme_event_2_in_new_scenario, probability_of_occurrence_of_extreme_event_1_in_a_past_scenario, probability_of_occurrence_of_extreme_event_2_in_a_past_scenario, probability_ratio_of_occurrence_of_the_compound_events_in_new_scenario, probability_ratio_of_occurrence_of_the_compound_events_in_a_past_scenario, event_1_name, event_2_name, time_period, scenario):
    
    """ Plot a map showing the probability of occurrence of a compound extreme climate event over the entire dataset time period
    
    Parameters
    ----------
    probability_of_occurrence_of_extreme_event_1_in_new_scenario : Xarray data array (probability of occurrence of extreme event 1 over the entire dataset time period in new scenario)
    probability_of_occurrence_of_extreme_event_2_in_new_scenario : Xarray data array (probability of occurrence of extreme event 2 over the entire dataset time period in new scenario)
    probability_of_occurrence_of_extreme_event_1_in_a_past_scenario : Xarray data array (probability_of_occurrence_of_extreme_event_1_in_a_past_scenario over the entire dataset time period in historical early industrial time period)
    probability_of_occurrence_of_extreme_event_2_in_a_past_scenario : Xarray data array (probability_of_occurrence_of_extreme_event_2_in_a_past_scenario over the entire dataset time period in historical early industrial time period)
    probability_ratio_of_occurrence_of_the_compound_events_in_new_scenario : Xarray data array
    probability_ratio_of_occurrence_of_the_compound_events_in_a_past_scenario : Xarray data array
    
    event_1_name, event_2_name : String (Extreme Events)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the Probability Ratio (PR) of joint occurrence of two extreme climate events assuming DEPENDENCE ONLY to compare the change in new scenario from the historical early industrial times
    
    """
    
    
    # Probability Ratio (PR) of occurrence: ratio between probability of occurrence of compound event in new situation (scenario) to the probability of occurrence in the reference situtaion (historical early industrial times)
    probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = ((probability_ratio_of_occurrence_of_the_compound_events_in_new_scenario/(probability_of_occurrence_of_extreme_event_1_in_new_scenario * probability_of_occurrence_of_extreme_event_2_in_new_scenario))/(probability_ratio_of_occurrence_of_the_compound_events_in_a_past_scenario/(probability_of_occurrence_of_extreme_event_1_in_a_past_scenario * probability_of_occurrence_of_extreme_event_2_in_a_past_scenario)))
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = True, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    # Set the legend limits   
    boundaries = [1, 2, 4, 6, 8, 10]
    n_colors = 6
    cmap = sns.color_palette('YlOrRd', n_colors=n_colors) # set color palette

    cmap = mcolors.ListedColormap(['cornflowerblue']+ [cmap[0]] + [cmap[1]]+ [cmap[2]]+ [cmap[3]] + [cmap[4]]+ ['darkred']) # manually select colors from palette
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=len(boundaries)+1, extend='both')
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
  
    formatter = ticker.FuncFormatter(lambda x, pos: str(Fraction(x).limit_denominator()) if x <= 1 else '{:.0f}'.format(x)) # convert the floats to fractions for those less than 1 and to integers for those above one in the color bar legend
    
    plt.imshow(probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = cmap, norm= norm)
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.265,0.9,'Comparing {} ({}) to 1861-1910'.format(time_period, scenario), fontsize = 9)
    
    # Add the title and legend to the figure and show the figure
    #plt.title('Probability Ratio of Joint Occurrence of {} and {} \n assuming only change in dependence \n'.format(event_1_name, event_2_name),fontsize=10) #Plot title
    

    plt.clim(0,10)
    
    #Plots the legend color bar
    colorbar = plt.colorbar(cbar, orientation = 'vertical', extend='both', extendfrac= 'auto', ticks=boundaries, format=formatter)
    colorbar.set_label(label = 'Probability Ratio', size = 14)
    
 
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    
    # Change this directory to save the plots to your desired directory
    #plt.savefig('C:/Users/dmuheki/OneDrive/PhD/Masters_thesis_paper/Ongoing_results/Probability Ratio of Joint Occurrence of {} and {} assuming only change in their dependence considering {} as the future secenario.pdf'.format(event_1_name, event_2_name, scenario), dpi = 300)
    
    plt.show()
    #plt.close()
    
    return probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only

#%%
def plot_percentage_contribution_of_probability_ratio_of_occurrence_of_an_extreme_event(percentage_contribution_of_ratio_of_occurrence_of_an_extreme_event, event_1_name, event_2_name, time_period, gcm, scenario):
    
    """ Plot a map showing the contribution of event to probability of joint occurrence of extreme climate events over the entire dataset time period
    
    Parameters
    ----------
    percentage_contribution_of_ratio_of_occurrence_of_an_extreme_event : Xarray data array 
    event_name : String (Extreme Events)
    gcm : String (Driving GCM)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the contribution of event to probability of joint occurrence of extreme climate events over the entire dataset time period
    
    """
    
    percentage_contribution_of_ratio_of_occurrence_of_an_extreme_event
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = True, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(percentage_contribution_of_ratio_of_occurrence_of_an_extreme_event, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = plt.cm.get_cmap('YlOrRd', 10))
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.265,0.9,'Comparing {} ({}) to 1861-1910'.format(time_period, scenario), fontsize = 9)
    plt.gcf().text(0.25,0.03,'{}'.format(gcm), fontsize= 10)
    
    # Add the title and legend to the figure and show the figure
    plt.title('Percentage contribution of occurrence of {} \n to the Joint Occurrence of {} and {} \n'.format(event_1_name, event_1_name, event_2_name),fontsize=10) #Plot title
    
    # discrete color bar legend
    #bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.colorbar( orientation = 'vertical', extend ='both').set_label(label = 'Contribution to Joint Occurrence (%)', size = 11) #Plots the legend color bar
    plt.clim(0,100)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    plt.show()
    #plt.close()
    
    return percentage_contribution_of_ratio_of_occurrence_of_an_extreme_event


#%%
def plot_percentage_contribution_of_probability_ratio_of_occurrence_of_two_extreme_events_in_dependence(percentage_contribution_of_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only, event_1_name, event_2_name, time_period, gcm, scenario):
    
    """ Plot a map showing the contribution of event to probability of joint occurrence of extreme climate events over the entire dataset time period
    
    Parameters
    ----------
    percentage_contribution_of_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only : Xarray data array 
    event_name : String (Extreme Events)
    gcm : String (Driving GCM)
    time_period: String
    scenario: String
    
    Returns
    -------
    Plot (Figure) showing the contribution of event to probability of joint occurrence of extreme climate events over the entire dataset time period
    
    """
    
    percentage_contribution_of_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only
    
    # Setting the projection of the map to cylindrical / Mercator
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add the background map to the plot
    #ax.stock_img()
    
    # Set the extent of the plot, in this case the East African Region
    ax.set_extent([19,54,-12,23])
    
    # Plot the coastlines along the continents on the map
    ax.coastlines(color='dimgrey', linewidth=0.7)
    
    # Plot features: lakes, rivers and boarders
    ax.add_feature(cfeature.LAKES, alpha =0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, facecolor ='lightgrey')
    #ax.add_feature(cfeature.BORDERS, linestyle=':')
    
     # Coordinates longitude and latitude ticks...These can be manipulated depending on study area chosen
    ax.set_xticks([20, 30, 40, 50], crs=ccrs.PlateCarree()) # 20E up to 50E
    ax.set_yticks([20, 10, 0, -10], crs=ccrs.PlateCarree()) # 20N up to 10S
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)  
    
    
    # Plot the gridlines for the coordinate system on the map
    #grid= ax.gridlines(draw_labels = True, dms = True )
    #grid.top_labels = False #Removes the grid labels from the top of the plot
    #grid.right_labels= False #Removes the grid labels from the right of the plot
    
    # Plot probability of occurrence of compound extreme events per location with the extent of the East African Region; Specified as (left, right, bottom, right)
    plt.imshow(percentage_contribution_of_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only, origin = 'upper' , extent=(18.25, 54.75, -12.75, 23.75), cmap = plt.cm.get_cmap('YlOrRd', 10))
    
    # Text outside the plot to display the time period & scenario (top-right) and the two Global Impact Models used (bottom left)
    plt.gcf().text(0.265,0.9,'Comparing {} ({}) to 1861-1910'.format(time_period, scenario), fontsize = 9)
    plt.gcf().text(0.25,0.03,'{}'.format(gcm), fontsize= 10)
    
    # Add the title and legend to the figure and show the figure
    plt.title('Contribution to Joint Occurrence assuming only change in dependence of \n {} and {} \n'.format(event_1_name, event_2_name),fontsize=10) #Plot title
    
    # discrete color bar legend
    #bounds = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.colorbar( orientation = 'vertical', extend='both').set_label(label = 'Contribution to Joint Occurrence (%)', size = 11) #Plots the legend color bar
    plt.clim(0,100)
    plt.xticks(fontsize=8, color = 'dimgrey') # color and size of longitude labels
    plt.yticks(fontsize=8, color = 'dimgrey') # color and size of latitude labels
    plt.show()
    #plt.close()
    
    return percentage_contribution_of_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only


#%% Function for plotting the box plot to compare the output from the different GCMS driving the different impact models
def boxplot_comparing_gcms(all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, event_1_name, event_2_name, gcms):
    '''
    

    Parameters
    ----------
    gcms_timeseries_of_joint_occurrence_of_compound_events : List of xarrays and gcm considered

    Returns
    -------
    Box plot.

    '''
    
    fig = plt.figure()
    
    graph = fig.add_gridspec(nrows = 1, ncols = len(all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events), wspace = 0) # create sub plots, with no space in between hence wspace = 0
    axes = graph.subplots(sharey =True) # to share the same y-axis
       
    colors = [(0.996, 0.89, 0.569), (0.996, 0.769, 0.31), (0.996, 0.6, 0.001), (0.851, 0.373, 0.0549), (0.6, 0.204, 0.016)] # color palette
    colors_palette = (sns.color_palette(colors)) # creating matplot color palette    

    
    for i in range(len(all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events)):       
        
        sns.set_palette(colors_palette)
        sns.color_palette(palette = colors_palette)
        plot = sns.boxplot(data = all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i], ax = axes[i], showmeans = True, showfliers = False, meanprops = {'marker':"o", 'markerfacecolor': "yellow", 'markersize': "5.0"})
        plot.set(xlabel = gcms [i])
        #plot.set(ylim=(0,60)) # limit y axis to fraction of 0.6
        
    for ax in fig.get_axes():

        ax.tick_params(bottom=True, labelbottom = False) # remove the x-axis labels
        
    for ax in axes.flat:
        ax.set(ylabel = 'Percentage of area')
        ax.label_outer() # avoid repetition of y axis label on all sub plots
    
    legend_elements = [Patch(facecolor= (0.996, 0.89, 0.569), edgecolor = (0.996, 0.89, 0.569), label ='early industrial'), Patch(facecolor= (0.996, 0.769, 0.31), edgecolor = (0.996, 0.769, 0.31), label ='historical'), Patch(facecolor= (0.996, 0.6, 0.001), edgecolor = (0.996, 0.6, 0.001), label ='rcp2.6'), Patch(facecolor= (0.851, 0.373, 0.0549), edgecolor = (0.851, 0.373, 0.0549), label ='rcp6.0'), Patch(facecolor= (0.6, 0.204, 0.016), edgecolor = (0.6, 0.204, 0.016), label ='rcp8.5')]
   
    plt.legend(handles = legend_elements, loc = 'upper left', bbox_to_anchor = (1,1), fontsize = 10)
    
    plt.gcf().text(0.60,0.9,'{} & {}'.format(event_1_name, event_2_name), fontsize = 10)
    
    fig.suptitle('Variation in the fraction of region affected by joint occurrence of {} and {} \n during 50-year periods demonstrated by multi-impact model ensembles \n'. format(event_1_name, event_2_name), y = 1.05)
    
    
    return all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events
    
    
#%% Function for sub plotting all the 15 box plots of the compound events
def all_boxplots(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, gcms):
    """
    

    Parameters
    ----------
    all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events 
    gcms : List

    Returns
    -------
    15 subplots of boxplots for all the compund event combinations.

    """
    
        
    fig = plt.figure(figsize=(8.27,11))
    outer = gridspec.GridSpec(5,3, wspace =0.3, hspace=0.25) #wspace is horizontal space between the subplots and hspace is the vertical space between the subplots
    
    
    legend_elements = [Patch(facecolor= (0.996, 0.89, 0.569), edgecolor = (0.996, 0.89, 0.569), label ='early industrial'), Patch(facecolor= (0.996, 0.769, 0.31), edgecolor = (0.996, 0.769, 0.31), label ='historical'), Patch(facecolor= (0.996, 0.6, 0.001), edgecolor = (0.996, 0.6, 0.001), label ='rcp2.6'), Patch(facecolor= (0.851, 0.373, 0.0549), edgecolor = (0.851, 0.373, 0.0549), label ='rcp6.0'), Patch(facecolor= (0.6, 0.204, 0.016), edgecolor = (0.6, 0.204, 0.016), label ='rcp8.5')]
    fig.legend(handles = legend_elements, bbox_to_anchor = (0.23,0.88), fontsize = 5) 

    colors = [(0.996, 0.89, 0.569), (0.996, 0.769, 0.31), (0.996, 0.6, 0.001), (0.851, 0.373, 0.0549), (0.6, 0.204, 0.016)] # color palette
    colors_palette = (sns.color_palette(colors)) # creating matplot color palette
    
    
    for i in range(len(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events)):
        
        all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events = all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][0]
        event_1_name = all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][1]
        event_2_name = all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][2]
        
        inner = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols = len(all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events), subplot_spec = outer[i], wspace=0) # create sub plots, with no space in between hence wspace = 0
        
        
        
        for j in range(len(all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events)):       
            
            
            if i == 0: ## Make first subplots of the 15, to have the certain y-axis limit 
                
                axes = plt.subplot(inner[j], ylim = (0,30))
            
            
            if i == 1: ## Make second subplots of the 15, to have the certain y-axis limit 
                               
                axes = plt.subplot(inner[j], ylim = (0,40))
            
            if i == 2: ## Make third subplot of the 15, to have a larger y-axis limit 
                
                axes = plt.subplot(inner[j], ylim = (0,70))
            
            if 2<i<5 : ## Make next subplots of the 15, to have a certain y-axis limit 
                
                axes = plt.subplot(inner[j], ylim = (0,20))
                
            if i == 5: ## Make next subplots of the 15, to have a certain y-axis limit 
                
                axes = plt.subplot(inner[j], ylim = (0,30))
                

            if i > 5:  ## Make next 12 subplots of the 15, to have a smaller y-axis limit
                
                axes = plt.subplot(inner[j], ylim = (0,10))                     
            
            axes.tick_params(axis ='y', labelsize = 7, pad = 2)
                                    
            sns.set_palette(colors_palette)
            sns.color_palette(palette = colors_palette)
            #sns.color_palette(palette = 'YlOrBr', as_cmap = True, n_colors=5)
            plot = sns.boxplot(data = all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[j], ax = axes, width = 0.5, showmeans = True, showfliers = False, linewidth = 1, boxprops = {'linestyle': ' '}, meanprops = {'marker':"o", 'markerfacecolor': "yellow", "markeredgecolor": "black", 'markersize': "3.0"})
            
            if i > 11: 
                # Label the driving GCMs per subplot
                plot.set_xlabel(gcms [j], rotation = 45, fontsize = 7)
                
            
            if j == 3:
                
                # Titles of compound events on all the subplots
                plt.gca().set_title('{} & {}'.format(event_1_name, event_2_name), loc= 'right',fontsize = 7.5)
                
            if j > 0:
                
                plt.gca().set_yticks([])
            
            if i == 0 or i == 3 or i == 6 or i == 9 or i == 12:
                
                axes.set(ylabel = 'Percentage of area')
                axes.yaxis.label.set_size(7)
                axes.label_outer() # avoid repetition of y axis label on all sub plots
            
            #TO EDIT THE YAXIS LINES / SEPARATORS
            if j == 1:
                
                #axes.spines.right.set_visible(False) # To hide the lines
                #axes.spines.left.set_visible(False)
                
                axes.spines.right.set_color('gray')
                axes.spines.left.set_color('gray')
            
            if j == 2:
                
                #axes.spines.left.set_visible(False) # To hide the lines
                #axes.spines.right.set_visible(False)
                
                axes.spines.right.set_color('gray')
                axes.spines.left.set_color('gray')
                
            
            if j == 3:
                #axes.spines.left.set_visible(False) # To hide the lines
                
                axes.spines.left.set_color('gray')
                
            
        for ax in fig.get_axes():

            ax.tick_params(bottom=True, labelbottom = False) # remove the x-axis labels
        

    fig.suptitle('Variation in the fraction of region affected by joint occurrence of compound events \n during 50-year periods demonstrated by multi-impact model ensembles', fontsize =10)    
    
    return all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events

#%% Function for plotting the box plot to compare the compaund events with all the output from the different GCMS driving the different impact models
def comparison_boxplot(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events):
    '''
    

    Parameters
    ----------
    all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events : List of xarrays 

    Returns
    -------
    Box plot.

    '''
    
    fig = plt.figure(figsize = (9,5.5))
    
    outer_box = gridspec.GridSpec(1,1) # outer box (bounding) all the sub plots
    outerax = fig.add_subplot(outer_box[0])
    outerax.tick_params(axis='both', which='both', length = 0, bottom=0, left=0, labelbottom=0, labelleft=0)  
    
    graph = fig.add_gridspec(nrows = 1, ncols = 15, wspace = 0.3) # create sub plots, with no space in between hence wspace = 0
    axes = graph.subplots(sharey =True) # to share the same y-axis
    
    
    colors = [(0.996, 0.89, 0.569), (0.996, 0.769, 0.31), (0.996, 0.6, 0.001), (0.851, 0.373, 0.0549), (0.6, 0.204, 0.016)] # color palette
    colors_palette = (sns.color_palette(colors)) # creating matplot color palette
    
    
    for i in range(len(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events)):       
        
        #all_data_considering_all_gcms = [[]]*(len(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][0][0]))
        
        all_data_considering_all_gcms = [[],[],[],[],[]]
        for gcm in range(len(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][0])):
            gcm_data = all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][0][gcm]
            for scenario in range(len(gcm_data)):
                data_per_gcm = gcm_data[scenario]
                all_data_considering_all_gcms[scenario].extend(data_per_gcm)
                

        
        sns.set_palette(colors_palette)
        sns.color_palette(palette = colors_palette)
        plot = sns.boxplot(data = all_data_considering_all_gcms, ax = axes[i], showmeans = True, showfliers = False, width = 0.8, linewidth = 0.8, boxprops = {'linestyle': ' '}, meanprops = {'marker':"o", 'markerfacecolor': "yellow", "markeredgecolor": "black", 'markersize': "3.0"})
        #plot.set(xlabel = '{} and {}'.format(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][1], all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][2]))
        plot.set(ylim=(0,62)) # limit y axis to fraction of 0.6
        
        
        # renaming event 1 with an acronym for plotting purposes
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][1] == 'River Floods':
            event_1_name = 'RF'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][1] == 'Droughts':
            event_1_name = 'DR'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][1] == 'Heatwaves':
            event_1_name = 'HW'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][1] == 'Crop Failures':
            event_1_name = 'CF'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][1] =='Wildfires':
            event_1_name = 'WF'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][1] == 'Tropical Cyclones':
            event_1_name ='TC'
        
        # renaming event 2 with an acronym for plotting purposes
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][2] == 'River Floods':
            event_2_name = 'RF'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][2] == 'Droughts':
            event_2_name = 'DR'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][2] == 'Heatwaves':
            event_2_name = 'HW'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][2] == 'Crop Failures':
            event_2_name = 'CF'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][2] =='Wildfires':
            event_2_name = 'WF'
        if all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events[i][2] == 'Tropical Cyclones':
            event_2_name ='TC'
        
        
        
        plot.set_xlabel('     {} & {}'.format(event_1_name, event_2_name), rotation = 315 , rotation_mode = 'anchor', labelpad = 0.0, loc = 'left', fontsize = 9)

        
        if 0<i<14:
            axes[i].spines.left.set_visible(False) # To hide the lines
            axes[i].spines.right.set_visible(False)
            axes[i].tick_params(axis='y', which='both', length = 0)
            
            xticks = axes[i].xaxis.get_major_ticks()
            xticks[0].set_visible(False)
            xticks[1].set_visible(False)
            xticks[3].set_visible(False)
            xticks[4].set_visible(False)
        
        if i == 0:
            axes[i].spines.right.set_visible(False)
            
            xticks = axes[i].xaxis.get_major_ticks()
            xticks[0].set_visible(False)
            xticks[1].set_visible(False)
            xticks[3].set_visible(False)
            xticks[4].set_visible(False)
            
        
        if i == 14:
            axes[i].spines.left.set_visible(False) # To hide the lines
            axes[i].tick_params(axis='y', which='both', length = 0)
            
            xticks = axes[i].xaxis.get_major_ticks()
            xticks[0].set_visible(False)
            xticks[1].set_visible(False)
            xticks[3].set_visible(False)
            xticks[4].set_visible(False)
        
        
    for ax in fig.get_axes():

        ax.tick_params(bottom=True, labelbottom = False) # remove the x-axis labels
        
    for ax in axes.flat:
        ax.set(ylabel = 'Percentage of area')
        ax.label_outer() # avoid repetition of y axis label on all sub plots
    
    
    
    legend_elements = [Patch(facecolor= (0.996, 0.89, 0.569), edgecolor = (0.996, 0.89, 0.569), label ='early industrial'), Patch(facecolor= (0.996, 0.769, 0.31), edgecolor = (0.996, 0.769, 0.31), label ='present day'), Patch(facecolor= (0.996, 0.6, 0.001), edgecolor = (0.996, 0.6, 0.001), label ='rcp2.6'), Patch(facecolor= (0.851, 0.373, 0.0549), edgecolor = (0.851, 0.373, 0.0549), label ='rcp6.0'), Patch(facecolor= (0.6, 0.204, 0.016), edgecolor = (0.6, 0.204, 0.016), label ='rcp8.5')]
    fig.legend(handles = legend_elements, bbox_to_anchor = (0.90,0.88), fontsize = 10, frameon=False)

    
    
    #fig.suptitle('Variation in the fraction of region affected by compound events \n in 50-year periods demonstrated by multi-impact model ensembles \n', y = 1.05)
    
    # Change this directory to save the plots to your desired directory
    #plt.savefig('C:/Users/dmuheki/OneDrive/PhD/Masters_thesis_paper/Ongoing_results/Variation in the fraction of region affected by compound events in 50-year periods demonstrated by multi-impact model ensembles.pdf', dpi = 300)
    
    plt.show()
    
    return all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events
    
    