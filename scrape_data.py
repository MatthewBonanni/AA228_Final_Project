import pdb
import copy
from tqdm import tqdm
import fastf1 as ff1
import numpy as np
import pandas as pd

years = [2021, 2022]
cols = [
    'Year',
    'TrackID',
    'DriverID',
    'TeamID',
    'LapNumber',
    'TyreLife',
    'CompoundID',
    'Stint',
    'Sector1Time',
    'Sector2Time',
    'Sector3Time',
    'SpeedI1',
    'SpeedI2',
    'SpeedFL',
    'SpeedST',
    'Position',
    'PitStop',
    'YellowFlag',
    'RedFlag',
    'SC',
    'VSC',
    'LapTime']
weather_cols = [
    'AirTemp',
    'Humidity',
    'Pressure',
    'Rainfall',
    'TrackTemp',
    'WindDirection',
    'WindSpeed']

track_ids = {}
i_track_id = 0

team_ids = {}
i_team_id = 0

driver_names = {}
i_driver_id = 0

compound_ids = {
    "SOFT" : 0,
    "MEDIUM" : 1,
    "HARD" : 2,
    "INTERMEDIATE" : 3,
    "WET" : 4}

# TODO - total DRS time
# TODO - save ids

data = pd.DataFrame()

for year in years:
    schedule = ff1.get_event_schedule(year = year, include_testing=False)
    for i_round in tqdm(range(len(schedule))):
        event_name = schedule['EventName'].iloc[i_round]
        if event_name in track_ids:
            track_id = track_ids[event_name]
        else:
            track_id = i_track_id
            track_ids[event_name] = i_track_id
            i_track_id += 1

        round_id = schedule['RoundNumber'].iloc[i_round]
        session = ff1.get_session(year, round_id, 'R')
        session.load(messages=False)
        lap_data = copy.deepcopy(session.laps)

        #lap_data = lap_data[lap_data['IsAccurate'] == True] # Might be removing important laps
        #Instead Filter out NaT Laps
        lap_data = lap_data[pd.isnull(lap_data['LapTime'])==False]
        # Filter out "UNKNOWN" compound
        lap_data = lap_data[lap_data["Compound"]!="UNKNOWN"]

        # Append and convert some columns
        lap_data['Year'] = year
        lap_data['TrackID'] = track_id
        lap_data['CompoundID'] = [int(compound_ids[comp]) for comp in lap_data['Compound']]
        lap_data['TrackStatusID'] = lap_data['TrackStatus'].to_numpy(dtype=int)
        lap_data['StartTime'] = lap_data['Time'] - lap_data['LapTime']
        lap_data["PitStop"] = np.zeros((len(lap_data),1))
        lap_data["PitStop"] = lap_data["PitStop"].where(pd.isnull(lap_data["PitOutTime"]),1)
        lap_data["YellowFlag"] = np.ones((len(lap_data),1))
        lap_data["YellowFlag"] = lap_data["YellowFlag"].where(['2' in ls for ls in lap_data["TrackStatus"]],0)
        lap_data["RedFlag"] = np.ones((len(lap_data),1))
        lap_data["RedFlag"] = lap_data["RedFlag"].where(['5' in ls for ls in lap_data["TrackStatus"]],0)
        lap_data["SC"] = np.ones((len(lap_data),1))
        lap_data["SC"] = lap_data["SC"].where(['4' in ls for ls in lap_data["TrackStatus"]],0)
        lap_data["VSC"] = np.ones((len(lap_data),1))
        lap_data["VSC"] = lap_data["VSC"].where(['6' in ls for ls in lap_data["TrackStatus"]],0)

        lap_data = lap_data[lap_data['RedFlag']==0]
            
        team_ids_arr = np.zeros(len(lap_data), dtype=int)
        for i_lap, team_name in enumerate(lap_data['Team']):
            if team_name in team_ids:
                team_id = team_ids[team_name]
            else:
                team_id = i_team_id
                team_ids[team_name] = i_team_id
                i_team_id += 1
            team_ids_arr[i_lap] = team_id
        lap_data['TeamID'] = team_ids_arr

        # collect Driver Names
        driver_ids_arr = np.zeros(len(lap_data), dtype=int)
        for i_lap, driver_name in enumerate(lap_data['Driver']):
            if driver_name in driver_names:
                driver_id = driver_names[driver_name]
            else:
                driver_id = i_driver_id
                driver_names[driver_name] = i_driver_id
                i_driver_id += 1
            driver_ids_arr[i_lap] = driver_id
        lap_data['DriverID'] = driver_ids_arr

        # Get weather data
        lap_data[weather_cols] = np.nan
        #breakpoint()
        for i_lap in range(len(lap_data)):
            time_delta = (session.weather_data['Time'] -
                        lap_data['StartTime'].iloc[i_lap])
            i_weather = np.argmin(np.abs(time_delta))
            #breakpoint()
            lap_data.loc[lap_data.index[i_lap],
                        weather_cols] = \
                            session.weather_data.loc[session.weather_data.index[i_weather],
                                                    weather_cols]

        if len(data) == 0:
            data = lap_data[cols + weather_cols]
        else:
            data = pd.concat([data, lap_data[cols + weather_cols]])

    # Save data
    data.to_hdf("data/f1_dataset.h5", key="data", format='fixed', mode='w')

# Save ID definitions
def write_ids(id_dict, name):
    with open(name, 'w') as f:
        for key, val in id_dict.items():
            f.write(str(val) + "," + key + "\n")

write_ids(track_ids, "data/track_ids.txt")
write_ids(team_ids, "data/team_ids.txt")
write_ids(compound_ids, "data/compound_ids.txt")
write_ids(driver_names, "data/driver_names.txt")

breakpoint()