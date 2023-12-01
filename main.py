import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

pipe = pickle.load(open('pipeline_nn.pkl', 'rb'))

teams = ['England', 'Pakistan', 'Sri Lanka', 'Australia', 'South Africa',
       'New Zealand', 'India', 'Zimbabwe', 'West Indies', 'Ireland',
       'Scotland', 'Kenya', 'Bangladesh', 'Afghanistan']

venues = ['New Wanderers Stadium', 'Sophia Gardens', 'Providence Stadium',
       'Kennington Oval', 'Sydney Cricket Ground', 'Edgbaston',
       'Brisbane Cricket Ground, Woolloongabba', 'Eden Park',
       'Melbourne Cricket Ground', "Queen's Park Oval, Port of Spain",
       'Shere Bangla National Stadium', 'Bellerive Oval',
       'Sheikh Zayed Stadium', 'Newlands', 'The Rose Bowl',
       'Riverside Ground', 'Saxton Oval', 'Kingsmead',
       'Warner Park, Basseterre', "National Cricket Stadium, St George's",
       'Trent Bridge', 'Western Australia Cricket Association Ground',
       'Punjab Cricket Association Stadium, Mohali',
       'Kensington Oval, Bridgetown', 'SuperSport Park',
       'Rangiri Dambulla International Stadium', 'Nehru Stadium',
       'R Premadasa Stadium', 'MA Chidambaram Stadium, Chepauk',
       'Adelaide Oval', 'Vidarbha Cricket Association Stadium, Jamtha',
       'Sir Vivian Richards Stadium, North Sound', 'Feroz Shah Kotla',
       'Eden Gardens', 'Sharjah Cricket Stadium', 'Sabina Park, Kingston',
       'Dubai International Cricket Stadium', 'University Oval',
       'Kinrara Academy Oval', 'Westpac Stadium', 'Seddon Park',
       'Headingley', 'Arnos Vale Ground, Kingstown',
       'Civil Service Cricket Club, Stormont', 'Old Trafford',
       'National Stadium', 'Pallekele International Cricket Stadium',
       "St George's Park", "Lord's", 'Sawai Mansingh Stadium',
       'Multan Cricket Stadium', 'Harare Sports Club', 'McLean Park',
       'Khan Shaheb Osman Ali Stadium', 'Hagley Oval', 'Gaddafi Stadium',
       'Mahinda Rajapaksa International Cricket Stadium, Sooriyawewa',
       'Zahur Ahmed Chowdhury Stadium', 'M Chinnaswamy Stadium',
       'Queens Sports Club', 'Wankhede Stadium',
       'Beausejour Stadium, Gros Islet', 'Manuka Oval',
       'Sardar Patel Stadium, Motera', 'Clontarf Cricket Club Ground',
       'Willowmoore Park']
# venues = ['Shere Bangla National Stadium', 'Harare Sports Club', 'R Premadasa Stadium']


st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)

with col1:
    bat_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowl_team = st.selectbox('Select bowling team', sorted(teams))

venue = st.selectbox('Select Venue', sorted(venues))
runs = st.number_input('Current Score')
overs = st.number_input('Overs done(works for over>5)')

col3, col4 = st.columns(2)
# col3 = st.columns(1)
# wickets = st.number_input('Wickets out')
with col3:
    wickets = st.number_input('Wickets out')

with col4:
    wickets_last_5 = st.number_input('Wickets in last five overs')


last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    # df['remaining_overs'] = 49.6 - df['overs']
    remaining_overs = 49.6 - overs
    # df['weight_overs'] = (df['remaining_overs'] / 49.6)
    weight_over = remaining_overs/49.6
    wickets_left = 10 - wickets
    # df['weight_wicket'] = (df['remaining_wickets'] / 10)
    weight_wicket = wickets_left/10
    # df['merge_weight'] = (df['remaining_overs'] * df['weight_overs']) + (df['remaining_wickets'] * df['weight_wicket'])
    merge_weight = (remaining_overs * weight_over) + (wickets_left * weight_wicket)
    balls_left = 50 - (overs*6)
    # balls_left = 204 - (overs*6)

    runrate = runs/overs
    input_df = pd.DataFrame(
     {'bat_team': [bat_team], 'bowl_team': [bowl_team],
      'venue': [venue], 'overs': [overs], 'runs': [runs],
      # 'balls_left': [balls_left],
      'wickets': [wickets_left],
      'runrate': [runrate], 'runs_last_5': [last_five], 'merge_weight': [merge_weight],
      'wickets_last_5': [wickets_last_5]})

    # input_df = pd.DataFrame(
    #     {'bat_team': [bat_team], 'bowl_team': [bowl_team],
    #      'venue': [venue], 'overs': [overs], 'runs': [runs],
    #      'balls_left': [balls_left], 'wickets_left': [wickets],
    #      'runrate': [runrate], 'runs_last_5': [last_five]})
    # st.header(input_df.info)
    # st.text(input_df.info)
    # columns_to_encode = ['bat_team', 'bowl_team', 'venue']
    # encoded_df = pd.get_dummies(data=input_df, columns=columns_to_encode)
    # st.text(encoded_df.shape)
    # encoded_result = pd.concat([input_df, encoded_df], axis=1)
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))

    st.title('main stats')
    st.text(balls_left)
    st.text(wickets_left)
    st.text(runrate)