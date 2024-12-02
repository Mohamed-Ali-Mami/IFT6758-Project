import streamlit as st
import pandas as pd
import numpy as np
import requests, json
import os
from milestone_3 import retrieve_and_save_single_game_data, download_data

st.set_page_config(layout="wide", page_title='Hockey Visualization App')

# session_state = dict of values that persist through any page refresh
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'm' not in st.session_state:
    st.session_state.m = None


def main():
    # ------------ BACKEND ------------
    def load_model(workspace, model, version):
        response = requests.post("http://serving:8080/download_model", data={
            "api_key": os.environ['COMET_API_KEY'],
            "workspace": workspace,
            "registry_name": model,
            "version": version
        })
        return response

    def predict_game(game_id, model):
        # functionality of appending if new events instead of predicting all events is implemented in app.py
        response = requests.post("http://serving:8080/predict", data={
            "game_id": game_id
        })
        if response.status_code == 500:
            return None

        pred_dict = json.loads(response.json()['predictions'])
        cols = []
        if model == 'distance_model':
            cols = ['Distance','goal_probability']
        elif model == 'angle_model':
            cols = ['Angle','goal_probability']
        elif model == 'both_model':
            cols = [ 'Angle','Distance','goal_probability']
        df_pred = pd.DataFrame(pred_dict, columns=cols)
        df_pred.rename(columns={'goal_probability':'xG'}, index=lambda x : f'Event  {x}', inplace=True)

        return df_pred


    # ------------ SIDEBAR ------------
    model_fields = ['Workspace', 'Model', 'Version']
    models = ['distance_model', 'angle_model', 'both_model']

    # store model info
    model_cfg = [None, None, None]
    for i in range(len(model_fields)):
        model_cfg[i] = st.sidebar.text_input(model_fields[i])

    # attempt to load model
    if st.sidebar.button('Get Model'):
        ws,m,v = model_cfg

        st.session_state.model_loaded = False
        if ws and m and v:
            if m not in models:  # don't call endpoint if invalid model
                st.sidebar.write('Invalid model name')
            else:
                response_lm = load_model(ws,m,v)
                if response_lm.status_code == 500:
                    st.sidebar.write(response_lm.json()['error'])
                else:
                    st.sidebar.write('Model successfully loaded')
                    st.session_state.model_loaded = True    # update to true for when prediction is asked for (user enters game id), we know if a model is successfully loaded or not
                    st.session_state.m = m      # only update model when it was succesfuly loaded (now if predict is clicked it will have the correct model name)
        else:
            st.sidebar.write('Enter all fields')    # don't call endpoint if fields missing

    # ------------ CONTENT ------------
    def create_period_events(df_pred, df_game):
        df_period_events = df_pred.copy()[['Team', 'Goal', 'xG']]

        df_period_events['Period'] = df_game['Period'].tolist()     # add period column (1,2 or 3)
        df_period_events['Goal'] = [1 if x == 'True' else 0 for x in df_period_events['Goal'].tolist()] # strings ('True' or 'False) to integers (1 or 0)
        df_period_events.rename(columns={'Goal': 'Goals'}, inplace=True)

        # group by team and then period (we sum Goals and xG + count number of occurences of each team = number of shots each team)
        df_period_events = df_period_events.groupby(['Team','Period']).agg({
            'xG': 'sum',
            'Goals': 'sum',
            'Team': 'count'
        }).rename(columns={'Team': 'Shots'}).reset_index()

        df_period_events['xG'] = df_period_events['xG'].round(2)    # round xG values

        df_period_events = pd.melt(df_period_events, id_vars=['Team', 'Period'], value_vars=['xG', 'Shots', 'Goals'], var_name='Metric', value_name='Value')    # reshape df

        df_period_events['Metric'] = pd.Categorical(df_period_events['Metric'], categories=['Shots','xG','Goals'], ordered=True)    # order the three metrics (makes most sense for Goals to be last)

        df_period_events = df_period_events.pivot_table(index=['Team', 'Metric'], columns='Period', values='Value', aggfunc='sum').reset_index() # reshape df

        # remove the team name in rows 2,3,5,6 to remove some cluster and make table cleaner
        clean_teams = df_period_events['Team'].tolist()
        clean_teams[1] = clean_teams[2] = ''
        if len(clean_teams) == 6:
            clean_teams[4] =  clean_teams[5] = ''
        df_period_events['Team'] = clean_teams


        # renaming columns + set index to the team name
        df_period_events.rename(columns={'Team': ' --- Team --- '}, inplace=True)
        df_period_events.set_index(' --- Team --- ', inplace=True)
        df_period_events.rename(columns={'Goal': 'Goals'}, inplace=True)
        df_period_events.rename(columns={'Metric': ''}, inplace=True)

        return df_period_events


    col1_layout ,col2_layout = st.columns([4,8])

    col2_layout.title('Hockey Visualization App')

    game_id = col2_layout.text_input('Game ID')
    if col2_layout.button('Ping Game'):
        if game_id:
            df_pred = None
            if st.session_state.model_loaded:
                df_pred = predict_game(game_id, st.session_state.m)     # get prediction

            if df_pred is None:
                col2_layout.write('Invalid game ID or no model loaded')
            else:
                df_game = download_data(game_id)[1]     # get more data on each shot event

                # insert empty net, goal and team columns in goal probability df
                df_pred.insert(0, 'Empty Net', ['True' if x else 'False' for x in df_game['EmptyNet'].tolist()])
                df_pred.insert(len(df_pred.columns)-1, 'Goal', ['True' if x else 'False' for x in df_game['is_Goal'].tolist()])
                df_pred.insert(0, 'Team', df_game['AttackingTeam'].tolist())

                # add more description
                df_pred.rename(columns={'Distance':'Distance to net (ft)'}, inplace=True)
                df_pred.rename(columns={'Angle': 'Angle to net(\u00b0)'}, inplace=True)

                # get overall game info (period/time, home/away teams, current score)
                game_data_json = retrieve_and_save_single_game_data(game_id)
                home_team, away_team = 'None', 'None'
                with open(game_data_json, 'r') as file:
                    game_data = json.load(file)
                    home_team = game_data['home']['name']['default']
                    away_team = game_data['away']['name']['default']

                period = game_data['period']
                time_remaining = game_data['time_remaining']

                col2_layout.markdown(f'### <div style="margin-top: 2%;">Game {game_id}: {home_team} vs {away_team}</div>', unsafe_allow_html=True)

                if game_data['game_state'] == 'OFF':   # if game is over
                    col2_layout.write('FINAL')
                else:
                    styled_period_text = f'<div style="font-size:20px;">Period {period} - {time_remaining} left</div>'
                    col2_layout.markdown(f'<div style="margin-top: 2%; margin-bottom:2%;">{styled_period_text}</div>', unsafe_allow_html=True)

                total_goals_home = game_data['home']['score']
                total_goals_away = game_data['away']['score']

                total_xg_home = df_pred[df_pred['Team'] == home_team]['xG'].sum()
                total_xg_away = df_pred[df_pred['Team'] == away_team]['xG'].sum()

                col21_content, col22_content = col2_layout.columns([7,5])
                col21_content.metric(f'{home_team} xG (actual)', f'{total_xg_home:.2f} ({total_goals_home})', f'{total_goals_home - total_xg_home:.1f}')
                col22_content.metric(f'{away_team} xG (actual)', f'{total_xg_away:.2f} ({total_goals_away})', f'{total_goals_away - total_xg_away:.1f}')

                df_period_events = create_period_events(df_pred, df_game)

                # shot metrics by period
                col1_layout.markdown('##### <div style="margin-top: 24.5em; margin-bottom:1.5%;">Shot stats by period</div>', unsafe_allow_html=True) # I WANT THIS LINE AND THE NEXT TO BE ALIGNED WITH THE TWO LINES AFTERWARDS
                col1_layout.write(df_period_events)
                col1_layout.write('With this new table, users can get a better idea of the flow of the game. For each team, we display the total number of shots, xG and goals in each period to give more context on the game and how the total xG was made up. The table was created by aggregating the shot events by team and period, the model used for prediction is the same as the one loaded.')

                # all shot events
                col2_layout.markdown(f'### <div style="margin-top: 3%; margin-bottom:1.5%;">Data used for predictions (and predictions)</div>', unsafe_allow_html=True)
                col2_layout.write(df_pred)
        else:
            col2_layout.markdown('### <div style="margin-top: 2%;">Enter Game ID to view stats</div>', unsafe_allow_html=True)  # don't display anything if invalid/no game id

    else:
        col2_layout.markdown('### <div style="margin-top: 2%;">Enter Game ID to view stats</div>', unsafe_allow_html=True)  # don't display anything if invalid/no game id

if __name__ == '__main__':
    main()
