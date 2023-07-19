# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:27:36 2021

@author: Derrick Muheki
"""

import os
import funcs as fn
import xarray as xr
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from datetime import datetime


# Start recording processing time
start_time = datetime.now()

#%% SETTING UP THE CURRENT WORKING DIRECTORY; for both the input and output folders
cwd = os.getcwd()

#%% LIST OF PARAMETERS FOR A NESTED LOOP


scenarios_of_datasets = ['historical','rcp26', 'rcp60', 'rcp85']

time_periods_of_datasets = ['1861-1910', '1956-2005', '2050-2099']
extreme_event_categories = ['floodedarea', 'driedarea', 'heatwavedarea', 'cropfailedarea', 'burntarea', 'tropicalcyclonedarea']


compound_events = [['floodedarea', 'burntarea'], ['floodedarea', 'heatwavedarea'], ['heatwavedarea', 'burntarea'], 
                   ['heatwavedarea', 'cropfailedarea'], ['driedarea', 'burntarea'], ['driedarea', 'heatwavedarea'],
                   ['cropfailedarea','burntarea'], ['floodedarea', 'driedarea'], ['floodedarea', 'cropfailedarea'],
                   ['driedarea', 'cropfailedarea'], ['heatwavedarea', 'tropicalcyclonedarea'], ['burntarea', 'tropicalcyclonedarea'],
                   ['floodedarea', 'tropicalcyclonedarea'], ['driedarea', 'tropicalcyclonedarea'], ['cropfailedarea', 'tropicalcyclonedarea']]



# list of bias-adjusted Global Climate Models available for all the Impact Models
gcms = ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']



#%% MASK: FOR UNIFORMITY IN THE PLOTS, A MASKING WILL BE DONE ON ALL THE EXTREME EVENTS DATA TO ENSURE NaN VALUES OVER THE OCEAN. 
#         This is because it was noticed that some events such as heatwaves and crop failure had zero values over the ocean instead of NaN values in some Global Impact Models 
#         FOR OUR CASE, ONLY FOR MASKING PURPOSES, FLOODS DATA FROM THE 'ORCHIDEE' IMPACT MODEL WILL BE USED TO MASK ALL THE OTHER EXTREME EVENTS. 

floods_data = os.path.join(cwd, 'floodedarea') #folder containing all flooded area data for all available impact models
floods_orchidee_dataset = os.path.join(floods_data, 'orchidee') # considering orchidee data

# For the period 1861 UNTIL 2005
one_file_historical_floods_orchidee_data =  os.path.join(floods_orchidee_dataset, 'orchidee_gfdl-esm2m_historical_floodedarea_global_annual_landarea_1861_2005.nc4')
start_year_of_historical_floods_data = fn.read_start_year(one_file_historical_floods_orchidee_data) #function to get the starting year of the data from file name
historical_floods_orchidee_data = fn.nc_read(one_file_historical_floods_orchidee_data, start_year_of_historical_floods_data, 'floodedarea', time_dim=True) # reading netcdf files based on variables
occurrence_of_historical_floods = fn.extreme_event(historical_floods_orchidee_data) # occurrence of floods..as a boolean...true or false. returns 1 where floods were recorded in that location during that year

# For the period 2006 UNTIL 2099
one_file_projected_floods_orchidee_data =  os.path.join(floods_orchidee_dataset, 'orchidee_gfdl-esm2m_rcp26_floodedarea_global_annual_landarea_2006_2099.nc4')
start_year_of_projected_floods_data = fn.read_start_year(one_file_projected_floods_orchidee_data) #function to get the starting year of the data from file name
projected_floods_orchidee_data = fn.nc_read(one_file_projected_floods_orchidee_data, start_year_of_projected_floods_data, 'floodedarea', time_dim=True) # reading netcdf files based on variables
occurrence_of_projected_floods = fn.extreme_event(projected_floods_orchidee_data) # occurrence of floods..as a boolean...true or false. returns 1 where floods were recorded in that location during that year


# mask for map uniformity purposes: to apply NaN values over the ocean
mask_for_historical_data = occurrence_of_historical_floods
mask_for_projected_data = occurrence_of_projected_floods

#%% FILE WITH ENTIRE GLOBE GRID CELL AREA in m2

entire_globe_grid_cell_areas_in_netcdf = 'entire_globe_grid_cell_areas.nc'

entire_globe_grid_cell_areas_in_xarray = (xr.open_dataset(entire_globe_grid_cell_areas_in_netcdf)).to_array()


#%% NESTED (FOR) LOOP FOR PLOTTING COMPOUND EXTREME EVENT OCCURRENCE, PROBABILITY OF JOINT OCCURRENCE AND MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS considering different combinations and rcp scenarios

# All 15 combinations of compound events and GCMS data on timeseries (50-year periods) of joint occurrence of compound events for all scenarios
all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events = []

for compound_event in compound_events:
    extreme_event_1 = compound_event[0]
    extreme_event_2 = compound_event[1]
        
    # Renaming of extreme events for plots/graphs
    extreme_event_1_name = fn.event_name(extreme_event_1)
    extreme_event_2_name = fn.event_name(extreme_event_2)
    
    
    # All GCMS data on timeseries (50-year periods) of joint occurrence of compound events for all scenarios
    all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events = []
    
    
    ## Extreme event occurrence per grid for a given scenario and given time period considering an ensemble of GCMs    
    
    average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models = [[],[],[],[],[]] # Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5
    average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models = [[],[],[],[],[]] # Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5
    average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models = [[],[],[],[],[]] # Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5

    
    # Maximum number of years with consecutive join occurrence of an extreme event pair considering all impact models and all their respective driving GCMs
    average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models = [[],[],[],[],[]] # Where order of list is early industrial, present day, rcp 2.6, rcp6.0 and rcp 8.5


    # Full set (50-year time periods) of timeseries of occurrence of two extreme events for all scenarios and all GCMS
    full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
    
    
    #
    gcms_timeseries_of_joint_occurrence_of_compound_events = []
    
    
    # List of bivariate distributions for constructing: One ellipse per GCM: 
    all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
    
    
    for gcm in gcms:
                
        
        #Considering all the impact models (with the same driving GCM)
        extreme_event_1_impact_models = fn.impact_model(extreme_event_1, gcm)
        extreme_event_2_impact_models = fn.impact_model(extreme_event_2, gcm)
        
        # timeseries (entire time periods) of joint occurrence of compound events for all scenarios
        timeseries_of_joint_occurrence_of_compound_events = []
        
        #timeseries (50-year periods) of joint occurrence of compound events for all scenarios
        timeseries_50_years_of_joint_occurrence_of_compound_events = []
        
        # Full set of timeseries (considering a pair/two extreme events) for all impact models friven by the same GCM
        gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = [] # (driven by same GCM) full set (50-year time periods) of timeseries of occurrence of two extreme events for all scenarios   
        
        
        for scenario in scenarios_of_datasets:
            
            extreme_event_1_dataset = fn.extreme_event_occurrence(extreme_event_1, extreme_event_1_impact_models, scenario)
            extreme_event_2_dataset = fn.extreme_event_occurrence(extreme_event_2, extreme_event_2_impact_models, scenario)
            
            
            if scenario == 'historical' :
                
                # EARLY INDUSTRIAL/ HISTORICAL / 50 YEARS / FROM 1861 UNTIL 1910
                
                all_impact_model_data_about_no_of_years_with_compound_events_from_1861_until_1910 = [] # List with total no. of years with compound events accross the multiple impact models driven by the same GCM
                all_impact_model_data_about_no_of_years_with_compound_events_from_1956_until_2005 = []
                
                all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910 = [] # List with maximum no. of years with consecutive compound events accross the multiple impact models driven by the same GCM
                all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005 = []
                
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_1910 = [] # List with timeseries of occurrence compound events accross the multiple impact models driven by the same GCM               
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_2005 = [] 
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1956_until_2005 = []
                
                #gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = [] # (driven by same GCM) full set (50-year time periods) of timeseries of occurrence of two extreme events for all scenarios
                
                all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910 = []
                all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910 = []
                
                all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005 = []
                all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005 = []
                
                
                early_industrial_time_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
                present_day_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
                
                
                for cross_category_impact_model_pair in itertools.product(extreme_event_1_dataset[0], extreme_event_2_dataset[0]):    # Iteration function to achieve comparison of one impact model of extreme event 1 with another impact model of extreme event 2, whereby both impact models are driven by the same GCM
                    
                    extreme_event_1_from_1861_until_1910_unmasked =  cross_category_impact_model_pair[0][0:50] # (UNMASKED) occurrence of event 1 considering one impact model 
                    extreme_event_2_from_1861_until_1910_unmasked =  cross_category_impact_model_pair[1][0:50] # (UNMASKED) occurrence of event 2 considering one impact model  

                    
                    extreme_event_1_from_1861_until_1910 = xr.where(np.isnan(mask_for_historical_data[0:50]), np.nan, extreme_event_1_from_1861_until_1910_unmasked) # (MASKED) occurrence of events considering one impact model
                    extreme_event_2_from_1861_until_1910 = xr.where(np.isnan(mask_for_historical_data[0:50]), np.nan, extreme_event_2_from_1861_until_1910_unmasked) # (MASKED) occurrence of events considering one impact model
                    
                    # full dataset from 1861 until 2005... to create timeseries data
                    extreme_event_1_from_1861_until_2005 = xr.where(np.isnan(mask_for_historical_data), np.nan, cross_category_impact_model_pair[0]) # (MASKED) occurrence of events considering one impact model
                    extreme_event_2_from_1861_until_2005 = xr.where(np.isnan(mask_for_historical_data), np.nan, cross_category_impact_model_pair[1]) # (MASKED) occurrence of events considering one impact model
                    
                    if len(cross_category_impact_model_pair[0]) == 0 or len(cross_category_impact_model_pair[1]) == 0: # checking for an empty array representing no data
                        print('No data available on occurrence of compound events for selected impact model during the period '+ time_periods_of_datasets[0] + '\n')
                    else:
                        
                        # OCCURRENCE OF COMPOUND EVENT FROM 1861 UNTIL 1910
                        occurrence_of_compound_events_from_1861_until_1910 = fn.compound_event_occurrence(extreme_event_1_from_1861_until_1910, extreme_event_2_from_1861_until_1910) #returns True for locations with occurence of compound events within same location in same year
                        
                        # TOTAL NO. OF YEARS WITH OCCURRENCE OF COMPOUND EVENTS FROM 1861 UNTIL 1910 (Annex to empty array above to later on determine the average total no. of years with compound events accross the multiple impact models driven by the same GCM)
                        no_of_years_with_compound_events_from_1861_until_1910 = fn.total_no_of_years_with_compound_event_occurrence(occurrence_of_compound_events_from_1861_until_1910)
                        all_impact_model_data_about_no_of_years_with_compound_events_from_1861_until_1910.append(no_of_years_with_compound_events_from_1861_until_1910) # Appended to the list above with total no. of years with compound events accross the multiple impact models driven by the same GCM
                        
                        # MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION
                        maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910 = fn.maximum_no_of_years_with_consecutive_compound_events(occurrence_of_compound_events_from_1861_until_1910)
                        all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910.append(maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910)
                        
                        #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT FROM 1861 UNTIL 1910 IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_1910 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_1861_until_1910, entire_globe_grid_cell_areas_in_xarray)
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_1910.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_1910) 
                        
                        #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT ACROSS THE FULL TIME SCALE IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        occurrence_of_compound_events_from_1861_until_2005 = fn.compound_event_occurrence(extreme_event_1_from_1861_until_2005, extreme_event_2_from_1861_until_2005) #returns True for locations with occurence of compound events within same location in same year
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_2005 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_1861_until_2005, entire_globe_grid_cell_areas_in_xarray)
                        
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_2005.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_2005) # Appended to the list above with timeseries of occurrence compound events accross the multiple impact models driven by the same GCM & #to_array() changes it from a Dataset to a Dataarray
                        
                        # COMPARING OCCURRENCE OF EXTREME EVENT 1 WITH EXTREME EVENT 2
                        timeseries_of_occurrence_of_extreme_event_1 = fn.timeseries_fraction_of_area_affected(extreme_event_1_from_1861_until_1910, entire_globe_grid_cell_areas_in_xarray)                                               
                        timeseries_of_occurrence_of_extreme_event_2 = fn.timeseries_fraction_of_area_affected(extreme_event_2_from_1861_until_1910, entire_globe_grid_cell_areas_in_xarray)
                        
                        # add timeseries array to tuple: to full_set_of_timeseries_of_occurrence_of_two_extreme_events ****** NOTE: This variable hasnt been used yet, to be used for bivariate distribution later
                        timeseries_of_occurrence_of_two_extreme_events = fn.set_of_timeseries_of_occurrence_of_two_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2)
                        early_industrial_time_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(timeseries_of_occurrence_of_two_extreme_events)
                        
                        # comparing timeseries of extreme event 1, event 2 and the joint occurrence of the two extremes 
                        comparison_plot = fn.plot_comparison_timeseries_fraction_of_area_affected_by_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2, timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], gcm, scenario)
                        
                        # probability of occurrence of individual extreme events 1 and 2 (past timeperiod --> reference period for probability ratios)
                        probability_of_occurrence_of_extreme_event_1_from_1861_until_1910 = fn.probability_of_occurrence_of_extreme_event(extreme_event_1_from_1861_until_1910)
                        all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910.append(probability_of_occurrence_of_extreme_event_1_from_1861_until_1910)
                        
                        probability_of_occurrence_of_extreme_event_2_from_1861_until_1910 = fn.probability_of_occurrence_of_extreme_event(extreme_event_2_from_1861_until_1910)
                        all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910.append(probability_of_occurrence_of_extreme_event_2_from_1861_until_1910)
                        
                
 
                
                    # PRESENT DAY/ HISTORICAL / 50 YEARS / FROM 1956 UNTIL 2005
                    
                    extreme_event_1_from_1956_until_2005_unmasked =  cross_category_impact_model_pair[0][95:] # (UNMASKED) occurrence of event 1 considering ensemble of gcms
                    extreme_event_2_from_1956_until_2005_unmasked =  cross_category_impact_model_pair[1][95:] # (UNMASKED) occurrence of event 2 considering ensemble of gcms
                    
                    extreme_event_1_from_1956_until_2005 = xr.where(np.isnan(mask_for_historical_data[95:]), np.nan, extreme_event_1_from_1956_until_2005_unmasked) # (MASKED) occurrence of events considering ensemble of gcms
                    extreme_event_2_from_1956_until_2005 = xr.where(np.isnan(mask_for_historical_data[95:]), np.nan, extreme_event_2_from_1956_until_2005_unmasked) # (MASKED) occurrence of events considering ensemble of gcms
                        
                    
                    if len(cross_category_impact_model_pair[0]) == 0 or len(cross_category_impact_model_pair[1]) == 0: # checking for an empty array representing no data
                        print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[1] + '\n')
                    else:
                        
                        # OCCURRENCE OF COMPOUND EVENT FROM 1956 UNTIL 2005
                        occurrence_of_compound_events_from_1956_until_2005 = fn.compound_event_occurrence(extreme_event_1_from_1956_until_2005, extreme_event_2_from_1956_until_2005) #returns True for locations with occurence of compound events within same location in same year
                        
                        
                        # TOTAL NO. OF YEARS WITH OCCURRENCE OF COMPOUND EVENTS FROM 1956 UNTIL 2005 (Annex to empty array above to later on determine the average total no. of years with compound events accross the multiple impact models driven by the same GCM)
                        no_of_years_with_compound_events_from_1956_until_2005 = fn.total_no_of_years_with_compound_event_occurrence(occurrence_of_compound_events_from_1956_until_2005)
                        all_impact_model_data_about_no_of_years_with_compound_events_from_1956_until_2005.append(no_of_years_with_compound_events_from_1956_until_2005) # Appended to the list above with total no. of years with compound events accross the multiple impact models driven by the same GCM
                        
                        # MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION
                        maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005 = fn.maximum_no_of_years_with_consecutive_compound_events(occurrence_of_compound_events_from_1956_until_2005)
                        all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005.append(maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005)
                        
                        #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT FROM 1956 UNTIL 2005 IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1956_until_2005 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_1956_until_2005, entire_globe_grid_cell_areas_in_xarray)
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1956_until_2005.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1956_until_2005)                     
                        
                        # COMPARING OCCURRENCE OF EXTREME EVENT 1 WITH EXTREME EVENT 2
                        timeseries_of_occurrence_of_extreme_event_1 = fn.timeseries_fraction_of_area_affected(extreme_event_1_from_1956_until_2005, entire_globe_grid_cell_areas_in_xarray)
                        timeseries_of_occurrence_of_extreme_event_2 = fn.timeseries_fraction_of_area_affected(extreme_event_2_from_1956_until_2005, entire_globe_grid_cell_areas_in_xarray)
                        
                        # add timeseries array to tuple: to full_set_of_timeseries_of_occurrence_of_two_extreme_events
                        timeseries_of_occurrence_of_two_extreme_events = fn.set_of_timeseries_of_occurrence_of_two_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2)
                        present_day_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(timeseries_of_occurrence_of_two_extreme_events)
                        
                        # comparing timeseries of extreme event 1, event 2 and the joint occurrence of the two extremes 
                        comparison_plot = fn.plot_comparison_timeseries_fraction_of_area_affected_by_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2, timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)

                        # probability of occurrence of individual extreme events 1 and 2 (during present day)
                        probability_of_occurrence_of_extreme_event_1_from_1956_until_2005 = fn.probability_of_occurrence_of_extreme_event(extreme_event_1_from_1956_until_2005)
                        all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005.append(probability_of_occurrence_of_extreme_event_1_from_1956_until_2005)
                        
                        probability_of_occurrence_of_extreme_event_2_from_1956_until_2005 = fn.probability_of_occurrence_of_extreme_event(extreme_event_2_from_1956_until_2005)
                        all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005.append(probability_of_occurrence_of_extreme_event_2_from_1956_until_2005)
                        
            
                
                
                
                # AREA AFFECTED BY COMPOUND EVENT FROM 1861 UNTIL 1910 IN SCENARIO
                if len(all_impact_model_data_about_no_of_years_with_compound_events_from_1861_until_1910) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[0] + '\n')
                else:   
                    
                    # THE AVERAGE TOTAL NO OF YEARS WITH COMPOUND EVENTS ACROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
                    average_no_of_years_with_compound_events_from_1861_until_1910 = xr.concat(all_impact_model_data_about_no_of_years_with_compound_events_from_1861_until_1910, dim='time').mean(dim='time', skipna= True)
                    plot_of_average_no_of_years_with_compound_events_from_1861_until_1910 = fn.plot_total_no_of_years_with_compound_event_occurrence(average_no_of_years_with_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], gcm, scenario)
                    
                    # THE AVERAGE PROBABILITY OF OCCURRENCE OF COMPOUND EVENTS FROM 1861 UNTIL 1910 ACCROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
                    average_probability_of_occurrence_of_the_compound_events_from_1861_until_1910 = fn.plot_probability_of_occurrence_of_compound_events(average_no_of_years_with_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], gcm, scenario)
                    average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[0].append(average_probability_of_occurrence_of_the_compound_events_from_1861_until_1910) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)                 
                                                    
                    # AVERAGE MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION FROM 1861 UNTIL 1910 (PLOTTED)
                    average_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910 = xr.concat(all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910, dim='time').mean(dim='time', skipna= True)
                    plot_of_average_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910 = fn.plot_maximum_no_of_years_with_consecutive_compound_events(average_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], gcm, scenario)
                    # Append to list containing all results from all impact models and their driving GCMs
                    average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[0].append(average_maximum_no_of_years_with_consecutive_compound_events_from_1861_until_1910)
                    
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE 50 YEAR TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_50_years_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_1910)
                    
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE ENTIRE TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1861_until_2005)
                
                
                # AVERAGE PROBABILITY OF OCCURRENCE OF INDIVIDUAL EXTREME EVENTS (DURING EARLY INDUSTRIAL PERIOD)
                average_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910, dim = 'models').mean(dim = 'models', skipna = True)
                average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[0].append(average_probability_of_occurrence_of_extreme_event_1_from_1861_until_1910_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                
                average_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910, dim = 'models').mean(dim = 'models', skipna = True)
                average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[0].append(average_probability_of_occurrence_of_extreme_event_2_from_1861_until_1910_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                
                
                
                
                # AREA AFFECTED BY COMPOUND EVENT FROM 1956 UNTIL 2005 IN SCENARIO
                if len(all_impact_model_data_about_no_of_years_with_compound_events_from_1956_until_2005) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[1] + '\n')
                else:
                    
                    # THE AVERAGE TOTAL NO OF YEARS WITH COMPOUND EVENTS ACROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
                    average_no_of_years_with_compound_events_from_1956_until_2005 = xr.concat(all_impact_model_data_about_no_of_years_with_compound_events_from_1956_until_2005, dim='time').mean(dim='time', skipna= True)
                    plot_of_average_no_of_years_with_compound_events_from_1956_until_2005 = fn.plot_total_no_of_years_with_compound_event_occurrence(average_no_of_years_with_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)
                    
                    # THE AVERAGE PROBABILITY OF OCCURRENCE OF COMPOUND EVENTS FROM 1956 UNTIL 2005 ACCROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
                    average_probability_of_occurrence_of_the_compound_events_from_1956_until_2005 = fn.plot_probability_of_occurrence_of_compound_events(average_no_of_years_with_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)
                    average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[1].append(average_probability_of_occurrence_of_the_compound_events_from_1956_until_2005) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)                 
                                                    
                    # AVERAGE MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION FROM 1956 UNTIL 2005 (PLOTTED)
                    average_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005 = xr.concat(all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005, dim='time').mean(dim='time', skipna= True)
                    plot_of_average_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005 = fn.plot_maximum_no_of_years_with_consecutive_compound_events(average_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)
                    # Append to list containing all results from all impact models and their driving GCMs
                    average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[1].append(average_maximum_no_of_years_with_consecutive_compound_events_from_1956_until_2005)
                    
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE 50 YEAR TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_50_years_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_1956_until_2005)
                    
                # AVERAGE PROBABILITY RATIO (PR) OF AVERAGE OCCURRENCE OF COMPOUND EVENTS FROM 1956 UNTIL 2005 COMPARED TO EVENTS FROM 1861 UNTIL 1910 (PLOTTED) ** average across all available impact models driven by the same GCM 
                average_probability_ratio_of_occurrence_of_the_compound_events_from_1956_until_2005 = fn.plot_probability_ratio_of_occurrence_of_compound_events(average_probability_of_occurrence_of_the_compound_events_from_1956_until_2005, average_probability_of_occurrence_of_the_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], gcm, scenario)
                    
                
                # Append list: Full set of timeseries (considering a pair/two extreme events) for all impact models driven by the same GCM 
                # Early industrial time
                gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(early_industrial_time_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events)    
                # Present age
                gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(present_day_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events)
                
                
                # AVERAGE PROBABILITY OF OCCURRENCE OF INDIVIDUAL EXTREME EVENTS (DURING PRESENT DAY)
                average_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005, dim = 'models').mean(dim = 'models', skipna = True)
                average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[1].append(average_probability_of_occurrence_of_extreme_event_1_from_1956_until_2005_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                
                average_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005, dim = 'models').mean(dim = 'models', skipna = True)
                average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[1].append(average_probability_of_occurrence_of_extreme_event_2_from_1956_until_2005_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                
                
                

                
            else:
                
                # Note: cropfailure events have no data for the rcp85 scenario: thus will be igonored in the rcp 85 scenario
                
                # END OF CENTURY / 50 YEARS / 2050 UNTIL 2099
                
                all_impact_model_data_about_no_of_years_with_compound_events_from_2050_until_2099 = [] # List with total no. of years with compound events accross the multiple impact models driven by the same GCM
                
                all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099 = [] # List with maximum no. of years with consecutive compound events accross the multiple impact models driven by the same GCM
                
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2006_until_2099 = [] # List with timeseries of occurrence of compound events accross the multiple impact models driven by the same GCM
                all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2050_until_2099 = []
                
                gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events_from_2050_until_2099 = [] # (driven by same GCM) full set (50-year time periods) of timeseries of occurrence of compound events accross the multiple impact models
                
                end_of_century_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events = []
                
                
                all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099 = []
                
                all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099 = []
                
                
                for cross_category_impact_model_pair in itertools.product(extreme_event_1_dataset[1], extreme_event_2_dataset[1]):  # Iteration function to achieve comparison of one impact model of extreme event 1 with another impact model of extreme event 2, whereby both impact models are driven by the same GCM
                    
                    extreme_event_1_from_2050_until_2099_unmasked =  cross_category_impact_model_pair[0][44:] # (UNMASKED) occurrence of event 1 considering one impact model 
                    extreme_event_2_from_2050_until_2099_unmasked =  cross_category_impact_model_pair[1][44:] # (UNMASKED) occurrence of event 2 considering one impact model
                    
                    extreme_event_1_from_2050_until_2099 = xr.where(np.isnan(mask_for_projected_data[44:]), np.nan, extreme_event_1_from_2050_until_2099_unmasked) # (MASKED) occurrence of events considering one impact model
                    extreme_event_2_from_2050_until_2099 = xr.where(np.isnan(mask_for_projected_data[44:]), np.nan, extreme_event_2_from_2050_until_2099_unmasked) # (MASKED) occurrence of events considering one impact model
                    
                    # full dataset from 1861 until 2005... to create timeseries data
                    extreme_event_1_from_2006_until_2099 = xr.where(np.isnan(mask_for_projected_data), np.nan, cross_category_impact_model_pair[0]) # (MASKED) occurrence of events considering one impact model
                    extreme_event_2_from_2006_until_2099 = xr.where(np.isnan(mask_for_projected_data), np.nan, cross_category_impact_model_pair[1]) # (MASKED) occurrence of events considering one impact model
                   
                    
                    if len(cross_category_impact_model_pair[0]) == 0 or len(cross_category_impact_model_pair[1]) == 0: # checking for an empty array representing no data
                        print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[2] + '\n')
                    else: 
                        
                        # OCCURRENCE OF COMPOUND EVENT FROM 2050 UNTIL 2099
                        occurrence_of_compound_events_from_2050_until_2099 = fn.compound_event_occurrence(extreme_event_1_from_2050_until_2099, extreme_event_2_from_2050_until_2099) #returns True for locations with occurence of compound events within same location in same year
                        
                        # TOTAL NO. OF YEARS WITH OCCURRENCE OF COMPOUND EVENTS FROM 2050 UNTIL 2099(Annex to empty array above to later on determine the average total no. of years with compound events accross the multiple impact models driven by the same GCM)
                        no_of_years_with_compound_events_from_2050_until_2099 = fn.total_no_of_years_with_compound_event_occurrence(occurrence_of_compound_events_from_2050_until_2099)                        
                        all_impact_model_data_about_no_of_years_with_compound_events_from_2050_until_2099.append(no_of_years_with_compound_events_from_2050_until_2099) # Appended to the list above with total no. of years with compound events accross the multiple impact models driven by the same GCM
                        
                        # MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION
                        maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099 = fn.maximum_no_of_years_with_consecutive_compound_events(occurrence_of_compound_events_from_2050_until_2099)
                        all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099.append(maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099)
                        
                        # TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT FROM 2050 UNTIL 2099 IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2050_until_2099 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_2050_until_2099, entire_globe_grid_cell_areas_in_xarray)
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2050_until_2099.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2050_until_2099)
                        
                        #TIMESERIES OF AFFECTED AREA BY COMPOUND EVENT ACROSS THE FULL TIME SCALE IN SCENARIO FOR EACH IMPACT MODEL IN THIS UNIQUE PAIR DRIVEN BY THE SAME GCM
                        occurrence_of_compound_events_from_2006_until_2099 = fn.compound_event_occurrence(extreme_event_1_from_2006_until_2099, extreme_event_2_from_2006_until_2099) #returns True for locations with occurence of compound events within same location in same year
                        timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2006_until_2099 = fn.timeseries_fraction_of_area_affected(occurrence_of_compound_events_from_2006_until_2099, entire_globe_grid_cell_areas_in_xarray)
                        
                        all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2006_until_2099.append(timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2006_until_2099) # Appended to the list above with timeseries of occurrence compound events accross the multiple impact models driven by the same GCM
                        
                        # COMPARING OCCURRENCE OF EXTREME EVENT 1 WITH EXTREME EVENT 2
                        timeseries_of_occurrence_of_extreme_event_1 = fn.timeseries_fraction_of_area_affected(extreme_event_1_from_2050_until_2099, entire_globe_grid_cell_areas_in_xarray)
                        timeseries_of_occurrence_of_extreme_event_2 = fn.timeseries_fraction_of_area_affected(extreme_event_2_from_2050_until_2099, entire_globe_grid_cell_areas_in_xarray)
                        
                        # add timeseries array to tuple: to full_set_of_timeseries_of_occurrence_of_two_extreme_events
                        timeseries_of_occurrence_of_two_extreme_events = fn.set_of_timeseries_of_occurrence_of_two_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2)
                        end_of_century_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(timeseries_of_occurrence_of_two_extreme_events)
                        
                        # comparing timeseries of extreme event 1, event 2 and the joint occurrence of the two extremes 
                        comparison_plot = fn.plot_comparison_timeseries_fraction_of_area_affected_by_extreme_events(timeseries_of_occurrence_of_extreme_event_1, timeseries_of_occurrence_of_extreme_event_2, timeseries_of_fraction_of_area_with_occurrence_of_compound_events_from_2050_until_2099, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)     
                        
                        # probability of occurrence of individual extreme events 1 and 2
                        probability_of_occurrence_of_extreme_event_1_from_2050_until_2099 = fn.probability_of_occurrence_of_extreme_event(extreme_event_1_from_2050_until_2099)
                        all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099.append(probability_of_occurrence_of_extreme_event_1_from_2050_until_2099)
                                                
                        probability_of_occurrence_of_extreme_event_2_from_2050_until_2099 = fn.probability_of_occurrence_of_extreme_event(extreme_event_2_from_2050_until_2099)
                        all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099.append(probability_of_occurrence_of_extreme_event_2_from_2050_until_2099)
                        
                        
                # AREA AFFECTED BY COMPOUND EVENT FROM 2050 UNTIL 2099 IN SCENARIO
                if len(all_impact_model_data_about_no_of_years_with_compound_events_from_2050_until_2099) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[2] + '\n')
                else:
                    
                    # THE AVERAGE TOTAL NO OF YEARS WITH COMPOUND EVENTS ACROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
                    average_no_of_years_with_compound_events_from_2050_until_2099 = xr.concat(all_impact_model_data_about_no_of_years_with_compound_events_from_2050_until_2099, dim='time').mean(dim='time', skipna= True)
                    plot_of_average_no_of_years_with_compound_events_from_2050_until_2099 = fn.plot_total_no_of_years_with_compound_event_occurrence(average_no_of_years_with_compound_events_from_2050_until_2099, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)
                    
                    # AVERAGE MAXIMUM NUMBER OF YEARS WITH CONSECUTIVE COMPOUND EVENTS IN SAME LOCATION FROM 2050 UNTIL 2099 (PLOTTED)
                    average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099 = xr.concat(all_impact_model_data_about_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099, dim='time').mean(dim='time', skipna= True)
                    plot_of_average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099 = fn.plot_maximum_no_of_years_with_consecutive_compound_events(average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)
                     
                    
                    # THE AVERAGE PROBABILITY OF OCCURRENCE OF COMPOUND EVENTS FROM 2050 UNTIL 2099 ACCROSS THE MULTIPLE IMPACT MODELS DRIVEN BY THE SAME GCM
                    average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099 = fn.plot_probability_of_occurrence_of_compound_events(average_no_of_years_with_compound_events_from_2050_until_2099, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)
                    if scenario == 'rcp26' :
                        average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[2].append(average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)    
                        average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[2].append(average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099)
                        
                    if scenario == 'rcp60' :
                        average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[3].append(average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)                 
                        average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[3].append(average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099)
                    
                    if scenario == 'rcp85' :
                        average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[4].append(average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)                 
                        average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[4].append(average_maximum_no_of_years_with_consecutive_compound_events_from_2050_until_2099)
                                                    
                    
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE 50 YEAR TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_50_years_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2050_until_2099)
                    
                    # FRACTION OF THE AREA AFFECTED BY COMPOUND EVENT ACROSS THE ENTIRE TIME SCALE IN SCENARIO (**list for all the impact models)
                    timeseries_of_joint_occurrence_of_compound_events.append(all_impact_model_data_timeseries_of_occurrence_of_compound_events_from_2006_until_2099)
                
                    
                # AVERAGE PROBABILITY RATIO (PR) OF AVERAGE OCCURRENCE OF COMPOUND EVENTS FROM 2050 UNTIL 2099 COMPARED TO EVENTS FROM 1861 UNTIL 1910 (PLOTTED) ** average across all available impact models driven by the same GCM 
                average_probability_ratio_of_occurrence_of_the_compound_events_from_2050_until_2099 = fn.plot_probability_ratio_of_occurrence_of_compound_events(average_probability_of_occurrence_of_the_compound_events_from_2050_until_2099, average_probability_of_occurrence_of_the_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], gcm, scenario)
                
                
                gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(end_of_century_gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events)
                            
                
                # AVERAGE PROBABILITY OF OCCURRENCE OF INDIVIDUAL EXTREME EVENTS (DURING THE END OF THE CENTURY)
                if len(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099) == 0 or len(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099) == 0: # checking for an empty array representing no data
                    print('No data available on occurrence of compound events for selected impact model and scenario during the period '+ time_periods_of_datasets[2] + '\n')
                else: 
                
                    if scenario == 'rcp26': 
                    
                        average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
                        average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[2].append(average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                        
                        average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
                        average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[2].append(average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                        
                    if scenario == 'rcp60' :
                        
                        average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
                        average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[3].append(average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                        
                        average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
                        average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[3].append(average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                        
                        
                    if scenario == 'rcp85' :
                        
                        average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
                        average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[4].append(average_probability_of_occurrence_of_extreme_event_1_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                        
                        average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm = xr.concat(all_impact_model_data_on_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099, dim = 'models').mean(dim = 'models', skipna = True)
                        average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[4].append(average_probability_of_occurrence_of_extreme_event_2_from_2050_until_2099_per_gcm) # Append list considering all GCMs, inorder to get the average probability across all GCMs (i.e. accross all impact models and their driving GCMs)
                        
                        
                                   
        # COMPARISON OF ALL THE SCENARIOS PER PAIR OF EXTREME EVENTS
        
        # Plot the timeseries showing the fraction of the total pixels affected by the joint occurrence of the compound events (set of two: extreme event 1 and 2) considering all impact models driven by the same GCM    
        plot_of_timeseries_of_joint_occurrence_of_compound_events = fn.plot_timeseries_fraction_of_area_affected_by_compound_events(timeseries_of_joint_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, gcm)
        
        # Append all 4 GCMS (50-year) timeseries of compound events
        all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events.append(timeseries_50_years_of_joint_occurrence_of_compound_events)
        
        # Append all 4 GCM SETS timeseries of occurrence of compound events (unique pair of extreme events). # To be used later for bivariate distribution
        gcms_timeseries_of_joint_occurrence_of_compound_events.append([timeseries_of_joint_occurrence_of_compound_events, gcm])
        
        
        
        # BIVARIATE DISTRUBUTION
        # Plot the pearson correlation coefficient considering fraction of total pixels affected by extreme event 1 and 2 in the same year
        plot_of_pearson_correlation_coefficient = fn.plot_correlation_with_spearmans_rank_correlation_coefficient(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events, extreme_event_1_name, extreme_event_2_name, gcm)
        
        
        # append list of bivariate distributions for all gcms to one list
        all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events.append(gcm_full_set_of_timeseries_of_occurrence_of_two_extreme_events)
        

        
        
        # ANY OTHER INDICES
        # ****** you can add any other desired indices here
          
        
    ## AVERAGE PROBABILITY RATIO
    
    # Probability Ratio for change in single extreme events (indiviually)   
    
    reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms = xr.concat(average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[0], dim = 'models').mean(dim = 'models', skipna = True)
    
    reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms = xr.concat(average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[0], dim = 'models').mean(dim = 'models', skipna = True)
    
    reference_period_average_probability_of_occurrence_of_compound_events = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[0], dim = 'models').mean(dim = 'models', skipna = True)
    
    for time_scenario in range(len(average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models)):
        
        scenario_considered = ['early-industrial', 'present day', 'rcp2.6', 'rcp6.0', 'rcp8.5']
        
        # Average probability of occurrence of compound events 
        # early industrial period
        plot_of_average_probability_of_occurrence_of_compund_events_from_1861_until_1910_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], scenario_considered[0])
        
        
        # Probability ratio
        if time_scenario != 0: # to avoid the early industrial period which is the reference period for the Probability Ratios for the other periods
            
            
            if len(average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[time_scenario]) == 0 or len(average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[time_scenario]) == 0 or len(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[time_scenario]) == 0 : # checking for an empty array representing no data
                print('No data available on occurrence of compound events for selected impact model and scenario during the period ')
            else: 
                
                # Average probability of occurrence of compound events 
                # present day
                average_probability_of_occurrence_of_compound_events_from_1956_until_2005 = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[1], dim = 'models').mean(dim = 'models', skipna = True)
                plot_of_average_probability_of_occurrence_of_compund_events_from_1956_until_2005_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(average_probability_of_occurrence_of_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], scenario_considered[1])
                # rcp26
                average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp26 = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[2], dim = 'models').mean(dim = 'models', skipna = True)
                plot_of_average_probability_of_occurrence_of_compund_events_from_2050_until_2099_under_rcp26_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp26, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[2])
                # rcp 60
                average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp60 = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[3], dim = 'models').mean(dim = 'models', skipna = True)
                plot_of_average_probability_of_occurrence_of_compund_events_from_2050_until_2099_under_rcp60_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp60, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[3])
                
                if len(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[4]) != 0:
                    #rcp 85
                    average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp85 = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[4], dim = 'models').mean(dim = 'models', skipna = True)
                    plot_of_average_probability_of_occurrence_of_compund_events_from_2050_until_2099_under_rcp85_considering_multi_model_ensembles = fn.plot_average_probability_of_occurrence_of_compound_events(average_probability_of_occurrence_of_compound_events_from_2050_until_2099_under_rcp85, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[4])
                   
                
                # PROBABILITY RATIO 
                
                # Due to Change in extreme event 1 only
                average_probability_of_occurrence_of_extreme_event_1_across_the_gcms = xr.concat(average_probability_of_occurrence_of_extreme_event_1_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
                if time_scenario == 1:
                    # for time period 1956_until_2005
                    probability_ratio_of_occurrence_of_extreme_event_1_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], scenario_considered[1])
                if time_scenario == 2:
                    # for time period 2050_until_2099
                    probability_ratio_of_occurrence_of_extreme_event_1_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[2])
                if time_scenario == 3:
                    # for time period 2050_until_2099
                    probability_ratio_of_occurrence_of_extreme_event_1_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[3])
                if time_scenario == 4:
                    # for time period 2050_until_2099
                    probability_ratio_of_occurrence_of_extreme_event_1_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[4])
                
            
                
                # Due to Change in extreme event 2 only
                average_probability_of_occurrence_of_extreme_event_2_across_the_gcms = xr.concat(average_probability_of_occurrence_of_extreme_event_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
                if time_scenario == 1:
                    # for time period 1956_until_2005
                    probability_ratio_of_occurrence_of_extreme_event_2_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, extreme_event_2_name, extreme_event_1_name, time_periods_of_datasets[1], scenario_considered[1])
                if time_scenario == 2:
                    # for time period 2050_until_2099
                    probability_ratio_of_occurrence_of_extreme_event_2_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, extreme_event_2_name, extreme_event_1_name, time_periods_of_datasets[2], scenario_considered[2])
                if time_scenario == 3:
                    # for time period 2050_until_2099
                    probability_ratio_of_occurrence_of_extreme_event_2_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, extreme_event_2_name, extreme_event_1_name, time_periods_of_datasets[2], scenario_considered[3])
                if time_scenario == 4:
                    # for time period 2050_until_2099
                    probability_ratio_of_occurrence_of_extreme_event_2_only = fn.plot_probability_ratio_of_occurrence_of_an_extreme_event_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, extreme_event_2_name, extreme_event_1_name, time_periods_of_datasets[2], scenario_considered[4])
                
                
                # Due to Change in dependence
                average_probability_of_occurrence_of_compound_events = xr.concat(average_probability_of_occurrence_of_the_compound_events_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
                if time_scenario == 1:
                    # for time period 1956_until_2005
                    probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = fn.plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, average_probability_of_occurrence_of_compound_events, reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], scenario_considered[1])
                if time_scenario == 2:
                    # for time period 2050_until_2099
                    probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = fn.plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, average_probability_of_occurrence_of_compound_events, reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[2])
                if time_scenario == 3:
                    # for time period 2050_until_2099
                    probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = fn.plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, average_probability_of_occurrence_of_compound_events, reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[3])
                if time_scenario == 4:
                    # for time period 2050_until_2099
                    probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only = fn.plot_probability_ratio_of_occurrence_of_two_extreme_events_assuming_dependence_only_considering_all_gcms(average_probability_of_occurrence_of_extreme_event_1_across_the_gcms, average_probability_of_occurrence_of_extreme_event_2_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_1_only_across_the_gcms, reference_period_average_probability_of_occurrence_of_extreme_event_2_only_across_the_gcms, average_probability_of_occurrence_of_compound_events, reference_period_average_probability_of_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[4])
                                    
                
            
    
    # AVERAGE MAXIMUM NUMBER OF YEARS WITH JOINT OCCURRENCE OF EXTREME EVENTS
    for time_scenario in range(len(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models)):
        
        scenario_considered = ['early-industrial', 'present day', 'rcp2.6', 'rcp6.0', 'rcp8.5']
        
        if len(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario]) == 0 : # checking for an empty array representing no data
            print('No data available on occurrence of compound events for selected impact model and scenario during the period')
        
        else:
            
            if time_scenario == 0:
                # for time period 1861_until_1910
                average_max_no_of_consecutive_years_with_compound_events_from_1861_until_1910 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
                plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_1861_until_1910, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[0], scenario_considered[time_scenario])
            
            if time_scenario == 1:
                # for time period 1956_until_2005
                average_max_no_of_consecutive_years_with_compound_events_from_1956_until_2005 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
                plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_1956_until_2005, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[1], scenario_considered[time_scenario])
            
            if time_scenario == 2:
                # for time period 2050_until_2099
                average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp26 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
                plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp26, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[time_scenario])
            
            if time_scenario == 3:
                # for time period 2050_until_2099
                average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp60 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
                plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp60, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[time_scenario])
            
            if time_scenario == 4:
                # for time period 2050_until_2099
                average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp85 = xr.concat(average_maximum_number_of_years_with_joint_occurrence_of_extreme_event_1_and_2_considering_all_gcms_and_impact_models[time_scenario], dim = 'models').mean(dim = 'models', skipna = True)
                plot_of_average_max_no_of_consecutive_years_with_compound_events = fn.plot_average_maximum_no_of_years_with_consecutive_compound_events_considering_all_impact_models_and_their_driving_gcms(average_max_no_of_consecutive_years_with_compound_events_from_2050_until_2099_under_rcp85, extreme_event_1_name, extreme_event_2_name, time_periods_of_datasets[2], scenario_considered[time_scenario])
            
            
    
    # BIVARIATE DISTRIBUTION CONSIDERING ALL GCMs
    
    bivariate_plot_considering_all_impact_models_per_gcm = fn.plot_correlation_with_spearmans_rank__correlation_coefficient_considering_scatter_points_from_all_impact_models(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events, extreme_event_1_name, extreme_event_2_name, gcms)
    
    #considering all impact models and all their driving GCMs per extreme event    
    combined_bivariate_plot_considering_all_impact_models_and_all_their_driving_gcms = fn.plot_correlation_with_spearmans_rank_correlation_coefficient_considering_scatter_points_from_all_impact_models_and_all_gcms(all_gcms_full_set_of_timeseries_of_occurrence_of_two_extreme_events, extreme_event_1_name, extreme_event_2_name)
    
    
    # BOX PLOT COMPARISON OF OCCURRENCE PER EXTREME EVENT PAIR
    
    box_plot_per_compound_event = fn.boxplot_comparing_gcms(all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name, gcms)
     
    all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events.append([all_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, extreme_event_1_name, extreme_event_2_name])
            
                
# COMPARISON PLOT FOR ALL THE 15 BOX PLOTS   
           
all_box_plots = fn.all_boxplots(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events, gcms)                
               
#considering all impact models and all their driving GCMs per extreme event  
new_box = fn.comparison_boxplot(all_compound_event_combinations_and_gcms_timeseries_50_years_of_joint_occurrence_of_compound_events)


# print total runtime of the code
end_time=datetime.now()
print('Processing duration: {}'.format(end_time - start_time))

print('*******************DONE************************ ')

