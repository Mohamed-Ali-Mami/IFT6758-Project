# -*- coding: utf-8 -*-
"""MILESTONE 3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-dUkxDPxly_Ff6i0ndQ2H0TSM5FARFmf

# Libraries importation
"""

import pandas as pd
import numpy as np
import requests
import json
import os
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

"""# Q1

## Function to save play-by-play json data for a single match by it's ID
"""

def retrieve_and_save_single_game_data(game_id:int):

      access_point = "https://api-web.nhle.com/v1/gamecenter/"
      data_directory = "data"

      if not os.path.exists(data_directory):
        os.makedirs(data_directory)

      # pass the game ID to the acces_point's URL
      url = f"{access_point}{game_id}/play-by-play"

      # Set the filename of the game
      filename = os.path.join(data_directory, f"game_{game_id}.json")

      # If the file exists then we skip the download
      if os.path.exists(filename):
          print(f"File {filename} succesfully loaded")
          # return filename

      # Check if we were able to access to the URL's data to download the play-by-play data
      response = requests.get(url)
      if response.status_code == 200:
          # retreive the required data
          playByPlay = response.json()['plays']
          home_team = response.json()['awayTeam']
          away_team = response.json()['homeTeam']
          period = response.json()['period']
          time_remaining = response.json()['clock']['timeRemaining']
          game_state = response.json()['gameState']

          # create and save the json informations into the file
          with open(filename, "w") as file:
              data_to_save = {"playByPlay": playByPlay, "home": home_team, "away": away_team, 'period': period, 'time_remaining': time_remaining, 'game_state': game_state}
              json.dump(data_to_save, file)
          print(f"data of match ID {game_id} are succesfully downloaded to {filename}")
      elif response.status_code == 404:
        print(f"404 NOT FOUND for game ID {game_id}: {response.status_code}")
      else:
          print(f"Error in game ID {game_id}: {response.status_code}")

      return filename

"""## Defining the pipeline and the classes"""
# ------------------------- calculate_angle FUNCTION ------------------------


def calculate_angle(row):
    # Coordinates of the shot
    x_shot, y_shot = row['XPoint'], row['YPoint']

    # Determine which net the shot is aimed at, based on RinkSide
    if row['RinkSide'] == 'right':
        x_net = -89  # X coordinate of the opponent's net if it's on the right side
    else:
        x_net = 89   # X coordinate of the opponent's net if it's on the left side or not specified

    y_net = 0  # Y coordinate of the opponent's net (middle of the net)

    # Calculate the angle between the shot and the middle of the net
    angle = math.degrees(math.atan2(y_shot - y_net, x_shot - x_net))

    # Adjust the angle so that it is negative if the shot comes from the right and positive if the shot comes from the left
    if row['RinkSide']!='right':
      if angle >= 90:
        angle -= 180
      elif angle <= -90:
        angle +=180

    return angle


# ------------------ PRE-PROCESSING PIPELINE -------------------------


class Angle_Calculator(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    X2['Angle'] = X2.apply(calculate_angle, axis=1)
    X2['Angle'] = X2['Angle'].round(0).astype(float)
    return X2


class create_PeriodSeconds_q4(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    # Convert "PeriodTime" to timedelta
    X2['PeriodTimeConverted'] = pd.to_timedelta('00:' + X2['PeriodTime'])

    # Calculate the total seconds
    periodSeconds = X2['PeriodTimeConverted'].dt.total_seconds()
    X2.insert(5, 'PeriodSeconds', periodSeconds)
    X2=X2.drop(columns=['PeriodTimeConverted'],axis=1)
    return X2

class CreatePreviousPeriodSeconds(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X2 = X.copy()
        X2['PreviousEventPeriodTimeConverted'] = pd.to_timedelta('00:' + X2['PreviousEventPeriodTime'])
        previousPeriodSeconds = X2['PreviousEventPeriodTimeConverted'].dt.total_seconds()
        X2.insert(X2.columns.get_loc('PreviousEventPeriodTime') + 1, 'PreviousPeriodSeconds', previousPeriodSeconds)
        X2 = X2.drop(columns=['PreviousEventPeriodTimeConverted'], axis=1)
        return X2


class add_rebound_column(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    X2['Rebound'] = X2['PreviousEvent'] == 'Shot'
    return X2

class add_AngleChangeOnRebound_column(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X2 = X.copy()

        # Initialize column for angle difference on rebounds
        X2['AngleChangeOnRebound'] = 0

        # Calculate the angle difference for shots that are rebounds
        for i in range(1, len(X2)):
            if X2.iloc[i]['Rebound'] == True:
                angle_diff = calculate_angle_difference(X2.iloc[i - 1]['Angle'], X2.iloc[i]['Angle'])
                X2.iloc[i, X2.columns.get_loc('AngleChangeOnRebound')] = round(angle_diff, 1).round(1)

        return X2

def calculate_angle_difference(angle1, angle2):
    # If the angles are on opposite sides, sum of absolute values
    if (angle1 <= 0 and angle2 >= 0) or (angle1 >= 0 and angle2 <= 0):
        return abs(angle1) + abs(angle2)
    # Otherwise, absolute difference between angles
    else:
        return abs(angle1 - angle2)



class add_PlaySpeed_Column(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X2 = X.copy()
        X2['PlaySpeed'] = 0.0

        for index, row in X2.iterrows():
            x1, y1 = row['PreviousX'], row['PreviousY']
            x2, y2 = row['XPoint'], row['YPoint']

            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            period1 = row['PreviousEventPeriod']
            period2 = row['Period']

            time1 = row['PreviousPeriodSeconds']
            time2 = row['PeriodSeconds']

            time_elapsed = time2 - time1
            if period1 != period2:
                time_elapsed += (period2 - period1) * 20 * 60

            if time_elapsed > 0:
                PlaySpeed = distance / time_elapsed
                X2.at[index, 'PlaySpeed'] = PlaySpeed
        X2['PlaySpeed'] = X2['PlaySpeed'].round(0).astype(float)
        return X2


class Rebound_Encode(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    X2["Rebound"] = X2["Rebound"].replace(True, 1).replace(False, 0).fillna(0).astype(int)
    return X2


class PeriodTime_Drop(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    X2 = X2.drop(columns=['PeriodTime','PreviousEventPeriodTime'])
    return X2


class ShotType_Encode(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    X2['ShotType'] = X2['ShotType'].fillna(X2['ShotType'].mode()[0])
    X2 = pd.concat([X2.drop(columns=["ShotType"],axis=1),pd.get_dummies(X2['ShotType']).add_prefix("ShotType_")], axis = 1)
    return X2

class PreviousEvent_Encode(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    X2 = pd.concat([X2.drop(columns=["PreviousEvent"],axis=1),pd.get_dummies(X2['PreviousEvent']).add_prefix("PreviousEvent_")], axis = 1)
    return X2

class ToFloat(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    X2 = X2.astype(float)
    return X2


class NansImpute(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    imputer= SimpleImputer(strategy='mean')
    X2['PlaySpeed']=imputer.fit_transform(X2[['PlaySpeed']])
    X2['Angle']=imputer.fit_transform(X2[['Angle']])
    X2['AngleChangeOnRebound']=imputer.fit_transform(X2[['AngleChangeOnRebound']])
    X2['AngleChangeOnRebound']=imputer.fit_transform(X2[['Angle']])
    X2['Distance']=imputer.fit_transform(X2[['Distance']])
    X2['PreviousX']=imputer.fit_transform(X2[['PreviousX']])
    X2['PreviousY']=imputer.fit_transform(X2[['PreviousY']])
    X2['XPoint']=imputer.fit_transform(X2[['XPoint']])
    X2['YPoint']=imputer.fit_transform(X2[['YPoint']])
    return X2


class drop_RinkSide_X_Y_columns(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    X2 = X.copy()
    X2 = X2.drop(["RinkSide","XPoint","YPoint"],axis=1)
    return X2


# creation of the pipeline
pipeline=Pipeline([
    ( ('Angle_Calculator', Angle_Calculator()) ),
    #( ('create_PeriodSeconds_q4', create_PeriodSeconds_q4()) ),
    ( ('drop_RinkSide_X_Y_columns', drop_RinkSide_X_Y_columns()) ),
    #( ('CreatePreviousPeriodSeconds', CreatePreviousPeriodSeconds()) ),
    #( ('add_rebound_column', add_rebound_column()) ),
    #( ('add_AngleChangeOnRebound_column', add_AngleChangeOnRebound_column()) ),
    #( ('add_Speed_Column', add_PlaySpeed_Column()) ),
    #( ('Rebound_Encode', Rebound_Encode()) ),
    #( ('PeriodTime_Drop', PeriodTime_Drop()) ),
    #( ('ShotType_Encode', ShotType_Encode()) ),
    #( ('PreviousEvent_Encode', PreviousEvent_Encode()) ),
    #( ('NansImpute', NansImpute()) ),
    #( ('ToFloat', ToFloat()) )
     ])

# calculate distance function
def calculate_distance(row):
  if np.isnan(row['XPoint']) or np.isnan(row['YPoint']):
      return np.nan  # replace with nans if we couldn't calculate the distance
  else:
      # Euclidien distance
    if row['RinkSide'] == "right":
      # oponents net coordinates
      X_net = -89  # X coordinate of the oponents net
      Y_net = 0   # Y coordinate of the oponents net
      X_distance = row['XPoint'] - X_net
      Y_distance = row['YPoint'] - Y_net
    else :
      # oponents net coordinates
      X_net = 89  # X coordinate of the oponents net
      Y_net = 0   # Y coordinate of the oponents net
      X_distance = row['XPoint'] - X_net
      Y_distance = row['YPoint'] - Y_net
  return int(math.sqrt(X_distance**2 + Y_distance**2))



def download_data(game_id):
    # Download the json file of the game by it's ID
    filename = retrieve_and_save_single_game_data(game_id)

    # Load the JSON file
    with open(filename, 'r') as file:
        match_data = json.load(file)

    # Access to the playbyplay sub directory
    playByPlays = match_data['playByPlay']

    # Initalise a list to store the shots&goals data
    dataframe = []

    # Get the home and away team names
    home_team_id = match_data['home']['id']
    home_team_name = match_data['home']['name']['default']
    away_team_id = match_data['away']['id']
    away_team_name = match_data['away']['name']['default']
    home_team_score = 0
    away_team_score = 0

    # Parse through plays
    for play in playByPlays:

      if play['typeDescKey'] in ['missed-shot', 'shot-on-goal','blocked-shot', 'goal'] : # We retreive the informations we need from the Shots plays
        is_g = 1 if play['typeDescKey'] == 'goal' else 0
        # Parse through all the playerTypes and only retreive shooters and goalies if they exist otherwise we replace them by a NaN
        x = play['details'].get('xCoord', np.nan)  # Check if 'x' exists and replace it with a NaN if it doesn't
        y = play['details'].get('yCoord', np.nan)  # Check if 'y' exists and replace it with a NaN if i
        situation_code = play.get('situationCode', np.nan)  # Check if 'situation_code' exists and replace it with a NaN if i
        period = play.get('period', np.nan)  # Check if period exists and replace it with a NaN if i
        shot_type = play['details'].get('shotType', np.nan)  # Check if ShotType exists and replace it with a NaN if i
        periodTime = play.get('timeInPeriod', np.nan)  # Check if timeInPeriod exists and replace it with a NaN if i


        # We check the team's event informations
        if play['details']['eventOwnerTeamId'] == home_team_id:
          rink_side = play.get('homeTeamDefendingSide', np.nan) # we replace the missing values by nan
          attacking_team = home_team_name
          defending_team = away_team_name
          home_team_score = home_team_score+1 if play['typeDescKey'] == 'goal' else home_team_score
        else:
          rs = play.get('homeTeamDefendingSide', np.nan)
          attacking_team = away_team_name
          defending_team = home_team_name
          away_team_score = away_team_score+1 if play['typeDescKey'] == 'goal' else away_team_score
          if rs == "right":
            rink_side = "left"

          elif rs == "left":
            rink_side = "right"

          else:
              rink_side= np.nan

        # Creation of the EmptyNet column (0 if empty 1 if not)
        if play['details']['eventOwnerTeamId'] == home_team_id:
          empty_net = 1 if situation_code[0] == 0 else 0
        else:
          empty_net = 1 if situation_code[3] == 0 else 0

        # We add to the list of shots the data extracted from the json file
        dataframe.append([
            #periodTime,
            rink_side,
            x,
            y,
            #shot_type,
            empty_net,
            #previous_x,
            #previous_y,
            previous_event,
            previous_period,previous_period_remaining_time,
            is_g,
            period,
            play['timeInPeriod'],
            play['timeRemaining'],
            attacking_team,
            away_team_score if attacking_team == away_team_name else home_team_score,
            away_team_score if attacking_team == home_team_name else home_team_score,
            defending_team
        ])


      if 'details' in play:
        previous_event = play.get('typeDescKey', np.nan)
        previous_x = play['details'].get('xCoord', np.nan)
        previous_y = play['details'].get('yCoord', np.nan)
        previous_period_remaining_time = play.get('timeRemaining', np.nan)  # Check if 'y' exists and replace it with a NaN if i
        previous_period = play.get('period', np.nan)  # Check if 'y' exists and replace it with a NaN if

    dataframe = pd.DataFrame(dataframe, columns=["RinkSide", "XPoint","YPoint", "EmptyNet", "PreviousEvent" , "PreviousEvent_Period", "PreviousEvent_PeriodRemainingTime", "is_Goal","Period","timeInPeriod","PeriodRemainingTime","AttackingTeam","AttackingTeamScore","DefendingTeamScore","DefendingTeam"])

    # Add the 'Distance' column to the DataFrame
    dataframe['Distance'] = dataframe.apply(calculate_distance, axis=1)

    # Creation of the final dataframe compatible with our trained models (same format and order)
    df_infos = pipeline.fit_transform(dataframe)[["EmptyNet", "Distance", "PreviousEvent" , "PreviousEvent_Period", "PreviousEvent_PeriodRemainingTime", "is_Goal", "Angle","Period","timeInPeriod","PeriodRemainingTime","AttackingTeam","AttackingTeamScore","DefendingTeamScore","DefendingTeam"]]
    df_models = pipeline.fit_transform(dataframe)[["EmptyNet", "Distance", "is_Goal", "Angle","AttackingTeam"]]
    return df_models,df_infos

if __name__ == '__main__':
    df = download_data(2022030412)
    df
