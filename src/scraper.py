import pandas as pd
import numpy as np
import os
from collections import Counter
import json
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from io import StringIO

MATCH_DIR = 'data/Matches'
PLAYER_DIR = 'data/Players'
SHOTS_DIR = 'data/Shots'

os.makedirs(MATCH_DIR, exist_ok=True)
os.makedirs(PLAYER_DIR, exist_ok=True)
os.makedirs(SHOTS_DIR, exist_ok=True)

config = pd.read_csv('config.txt',sep='=').set_index('Type')
league = config.loc['League']['Value']
season = config.loc['Season']['Value']
season_name = config.loc['SeasonName']['Value']
playoff_date = config.loc['PlayoffDate']['Value']

def make_columns_unique(columns):
    counts = {}
    new_cols = []
    for col in columns:
        if col not in counts:
            counts[col] = 0
            new_cols.append(col)
        else:
            counts[col] += 1
            new_cols.append(f"{col}_{counts[col]}")
    return new_cols

def get_website(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)
        content = page.content()
        browser.close()
        soup = BeautifulSoup(content,'html.parser')
        site_json = json.loads(soup.text)
        return site_json

def get_matches(league,season):
    matches = list()
    i = 0
    while 1:
        url = 'https://api.sofascore.com/api/v1/unique-tournament/'+ league + '/season/' + season + '/events/last/' + str(i)
        try:
            returned = get_website(url)['events']
            matches += returned
            i += 1
        except:
            break
    
    i = 0
    while 1:
        url = 'https://api.sofascore.com/api/v1/unique-tournament/'+ league + '/season/' + season + '/events/next/' + str(i)
        try: 
            returned = get_website(url)['events']
            matches += returned
            i += 1
        except:
            break
    return matches

def create_results(league,season):
    results_list = []
    schedule_list = []
    results = get_matches(league,season)
    for match in results:
        match_id = match['id']
        status = match['status']['type']
        if (status == 'notstarted') | (status == 'inprogress'):
            game_time = match['startTimestamp']
            temp_hometeamname = match['homeTeam']['name']
            temp_hometeamid = match['homeTeam']['id']
            temp_hometeamcolorsprimary = match['homeTeam']['teamColors']['primary']
            temp_hometeamcolorssecondary = match['homeTeam']['teamColors']['secondary']
            temp_hometeamcolorstext = match['homeTeam']['teamColors']['text']
            temp_awayteamname = match['awayTeam']['name']
            temp_awayteamid = match['awayTeam']['id']
            temp_awayteamcolorsprimary = match['awayTeam']['teamColors']['primary']
            temp_awayteamcolorssecondary = match['awayTeam']['teamColors']['secondary']
            temp_awayteamcolorstext = match['awayTeam']['teamColors']['text'] 
            schedule_list.append([league,match_id,game_time,temp_hometeamname,temp_hometeamid,temp_hometeamcolorsprimary,temp_hometeamcolorssecondary,
                                  temp_hometeamcolorstext,temp_awayteamname,temp_awayteamid,temp_awayteamcolorsprimary,temp_awayteamcolorssecondary,temp_awayteamcolorstext])             
        else:
            results_list.append([match_id,status])
    results_list = pd.DataFrame(results_list,columns=['match_id','status'])
    results_list = results_list[results_list.status == 'finished'].match_id.astype('str').values    
    schedule_list = pd.DataFrame(schedule_list,columns=['league','match_id','game_date','home','home_id','home_primary','home_secondary','home_text','away','away_id','away_primary','away_secondary','away_text'])
    schedule_list['season'] = pd.to_datetime(schedule_list.game_date,unit='s').dt.year - 2000
    schedule_list.loc[pd.to_datetime(schedule_list.game_date,unit='s') >= pd.to_datetime(playoff_date),'type'] = 'playoff'
    schedule_list.loc[pd.to_datetime(schedule_list.game_date,unit='s') < pd.to_datetime(playoff_date),'type'] = 'regular'
    schedule_list.drop_duplicates().to_csv('data/Schedule.csv')
    return results_list

def get_match(match):
    url = 'https://api.sofascore.com/api/v1/event/' + str(match)
    returned = get_website(url)['event']
    return returned

def get_match_xg(match):
    url = 'https://api.sofascore.com/api/v1/event/' + str(match) + '/statistics'
    returned = get_website(url)
    return returned

def extract_match_summaries(match):
    temp_match = get_match(match)
    temp_id = match
    temp_gametime = temp_match['startTimestamp']
    temp_hometeamname = temp_match['homeTeam']['name']
    temp_hometeamid = temp_match['homeTeam']['id']
    temp_hometeamcolorsprimary = temp_match['homeTeam']['teamColors']['primary']
    temp_hometeamcolorssecondary = temp_match['homeTeam']['teamColors']['secondary']
    temp_hometeamcolorstext = temp_match['homeTeam']['teamColors']['text']
    temp_awayteamname = temp_match['awayTeam']['name']
    temp_awayteamid = temp_match['awayTeam']['id']
    temp_awayteamcolorsprimary = temp_match['awayTeam']['teamColors']['primary']
    temp_awayteamcolorssecondary = temp_match['awayTeam']['teamColors']['secondary']
    temp_awayteamcolorstext = temp_match['awayTeam']['teamColors']['text']
    try:
        temp_hometeamscore = temp_match['homeScore']['current']
        temp_awayteamscore = temp_match['awayScore']['current']
    except:
        temp_hometeamscore = np.nan
        temp_awayteamscore = np.nan

    temp_xg = get_match_xg(match)
    try:
        if temp_xg['statistics'][0]['groups'][0]['statisticsItems'][1]['key'] == 'expectedGoals':
            home_xg = temp_xg['statistics'][0]['groups'][0]['statisticsItems'][1]['homeValue']
            away_xg = temp_xg['statistics'][0]['groups'][0]['statisticsItems'][1]['awayValue']    
        else:
            home_xg = np.nan
            away_xg = np.nan
    except:
        home_xg = np.nan
        away_xg = np.nan

    try:
        temp_status = temp_match['hasEventPlayerStatistics']
    except:
        temp_status = 'Missing'
    
    temp_df = [[temp_id,temp_gametime,temp_hometeamname,temp_hometeamid,temp_hometeamcolorsprimary,temp_hometeamcolorssecondary,temp_hometeamcolorstext,
               temp_awayteamname,temp_awayteamid,temp_awayteamcolorsprimary,temp_awayteamcolorssecondary,temp_awayteamcolorstext,temp_hometeamscore,
               temp_awayteamscore,temp_status,home_xg,away_xg]]
    temp_df = pd.DataFrame(temp_df,columns=['match_id','game_date','home','home_id','home_primary','home_secondary','home_text','away',
                                                      'away_id','away_primary','away_secondary','away_text','home_score','away_score','status','home_xg','away_xg'])
    temp_df.loc[pd.to_datetime(temp_df.game_date,unit='s') >= pd.to_datetime(playoff_date),'type'] = 'playoff'
    temp_df.loc[pd.to_datetime(temp_df.game_date,unit='s') < pd.to_datetime(playoff_date),'type'] = 'regular'
    temp_df.reset_index(drop=True).to_feather(MATCH_DIR + '/' + temp_id + '.ftr')

def get_stats(match):
    url = 'https://api.sofascore.com/api/v1/event/' + str(match) + '/lineups'
    response = get_website(url)

    home_players = response['home']['players']
    away_players = response['away']['players']
    for p in home_players:
        p["teamLoc"] = 'Home'
    for p in away_players:
        p["teamLoc"] = 'Away'
    players = home_players + away_players
                    
    temp = pd.DataFrame(players)
    columns = list()
    for c in temp.columns:
        if isinstance(temp.loc[0, c], dict):
            columns.append(temp[c].apply(pd.Series, dtype=object))
        else:
            columns.append(temp[c])
    df = pd.concat(columns, axis=1)
    df.columns = make_columns_unique(df.columns)
    df.reset_index(drop=True).to_feather(PLAYER_DIR + '/' + match + '.ftr')

def get_shots(match):
    url = 'https://api.sofascore.com/api/v1/event/' + str(match) + '/shotmap'
    response = get_website(url)
    try:
        shotmap = response['shotmap']
        temp_shotdf = []
        for i in range(0,len(shotmap)):
            temp_shot = shotmap[i]
            player = temp_shot['player']['id']
            isHome = temp_shot['isHome']
            try:
                xg = temp_shot['xg']
            except:
                xg = np.nan
            try:
                xgot = temp_shot['xgot']
            except:
                xgot = np.nan
            timeSeconds = temp_shot['timeSeconds']
            shotType = temp_shot['shotType']
            try:
                goalType = temp_shot['goalType']
            except:
                goalType = np.nan
            situation = temp_shot['situation']
            playerCoordinates = temp_shot['playerCoordinates']
            bodyPart = temp_shot['bodyPart']
            goalMouthLocation = temp_shot['goalMouthLocation']
            goalMouthCoordinates = temp_shot['goalMouthCoordinates']
            try:
                blockCoordinates = temp_shot['blockCoordinates']
            except:
                blockCoordinates = np.nan
            incidentType = temp_shot['incidentType']
            temp_shotdf.append([match,temp_shot,player,isHome,xg,xgot,timeSeconds,shotType,situation,playerCoordinates,bodyPart,goalMouthLocation,
                                goalMouthCoordinates,blockCoordinates,incidentType])
        df = pd.DataFrame(temp_shotdf,columns = ['match','temp_shot','player','isHome','xg','xgot','timeSeconds','shotType','situation','playerCoordinates',
                                                'bodyPart','goalMouthLocation','goalMouthCoordinates','blockCoordinates','incidentType'])
        df.reset_index(drop=True).to_feather(SHOTS_DIR + '/' + match + '.ftr')
    except:
        print(url)

def summarize_matches(league,match_summaries):
    summary_data = []
    for i in match_summaries:
        temp = pd.read_feather(MATCH_DIR+'/'+str(i)+'.ftr')
        summary_data.append(temp)

    summary_data = pd.concat(summary_data)
    summary_data['league'] = league
    summary_data['season'] = pd.to_datetime(summary_data.game_date,unit='s').dt.year - 2000
    summary_data.loc[pd.to_dateteim(summary_data.game_date,unit='s') >= pd.to_datetime(playoff_date),'type'] = 'playoff'
    summary_data.loc[pd.to_dateteim(summary_data.game_date,unit='s') < pd.to_datetime(playoff_date),'type'] = 'regular'    
    return summary_data

class ColumnRenamer:
    def __init__(self, separator=None):
        self.counter = Counter()
        self.sep = separator

    def __call__(self, x):
        index = self.counter[x]  # Counter returns 0 for missing elements
        self.counter[x] = index + 1  # Uses something like `setdefault`
        return f'{x}{self.sep if self.sep and index else ""}{index if index else ""}'

def summarize_players(player_stats):
    player_data = {}
    for i in player_stats:
        temp = pd.read_feather(PLAYER_DIR + '/' + str(i) + '.ftr')
        temp['match_id'] = i
        player_data[i] = temp
    cols = ['teamLoc','name','id','position','position_1','proposedMarketValueRaw','substitute','minutesPlayed','rating',
                                          'goalAssist','goals','penaltyConceded','expectedGoals','expectedAssists','goalsPrevented','ownGoals','saves']
    player_data = pd.concat(player_data).reindex(columns=cols)
    player_data = player_data.reset_index().rename(columns={'level_0':'match_id'})
    return player_data

def summarize_shots(shots):
    shot_data = []
    for i in shots:
        temp = pd.read_feather(SHOTS_DIR + '/' + str(i) + '.ftr')
        temp['match_id'] = i
        shot_data.append(temp)
    cols = ['match_id','isHome','player','shotType','goalType','situation','playerCoordinates','bodyPart','goalMouthLocation',
                                      'goalMouthCoordinates','xg','xgot','id','time','timeSeconds','draw','periodTimeSeconds','blockCoordinates',
                                      'incidentType']
    shot_data = pd.concat(shot_data).reindex(columns=cols)
    shot_data[['xg','xgot']] = shot_data[['xg','xgot']].fillna(0)
    shot_data.loc[shot_data.goalType == 'own','own'] = 1
    shot_data.own = shot_data.own.fillna(0)
    return shot_data

def finalize_players(summary_data,player_data):
    home_summary = summary_data[['league','season','match_id','game_date','home','away','home_primary','home_secondary','home_text','home_score',
                             'away_score','home_xg','away_xg']].merge(player_data[player_data.teamLoc == 'Home']).rename(
    columns={'home':'team','away':'opponent','home_primary':'primary','home_secondary':'secondary','home_text':'text','home_score':'for',
             'away_score':'against','home_xg':'xg_for','away_xg':'xg_against'}).reset_index(drop=True)
    away_summary = summary_data[['league','season','match_id','game_date','away','home','away_primary','away_secondary','away_text','away_score',
                             'home_score','home_xg','away_xg']].merge(player_data[player_data.teamLoc == 'Away']).rename(
    columns={'away':'team','home':'opponent','away_primary':'primary','away_secondary':'secondary','away_text':'text','away_score':'for',
             'home_score':'against','home_xg':'xg_against','away_xg':'xg_for'}).reset_index(drop=True)
    player_stats = pd.concat((home_summary,away_summary))
    player_stats.game_date = pd.to_datetime(player_stats.game_date,unit='s').dt.date
    player_stats.to_feather('data/player_stats.ftr')

def finalize_shots(summary_data,shot_data):
    home_summary = summary_data[['league','season','match_id','game_date','home','away','home_primary','home_secondary','home_text','home_score',
                             'away_score','home_xg','away_xg']].merge(shot_data[shot_data.isHome == True]).rename(
    columns={'home':'team','away':'opponent','home_primary':'primary','home_secondary':'secondary','home_text':'text','home_score':'for',
             'away_score':'against','home_xg':'xg_for','away_xg':'xg_against'}).reset_index(drop=True)
    away_summary = summary_data[['league','season','match_id','game_date','away','home','away_primary','away_secondary','away_text','away_score',
                                 'home_score','home_xg','away_xg']].merge(shot_data[shot_data.isHome == False]).rename(
        columns={'away':'team','home':'opponent','away_primary':'primary','away_secondary':'secondary','away_text':'text','away_score':'for',
                 'home_score':'against','home_xg':'xg_against','away_xg':'xg_for'}).reset_index(drop=True)
    shots_stats = pd.concat((home_summary,away_summary))
    shots_stats.game_date = pd.to_datetime(shots_stats.game_date,unit='s').dt.date
    shots_stats.to_feather('data/shot_stats.ftr')

def finalize_matches(summary_data,shot_data):
    shot_agg = shot_data.groupby(['match_id','isHome']).xg.apply(list).reset_index().merge(
    shot_data.groupby(['match_id','isHome']).xgot.apply(list).reset_index())
    og_agg = shot_data.groupby(['match_id','isHome']).own.sum().reset_index()

    match_stats = summary_data.merge(shot_agg[shot_agg.isHome == True].drop(columns='isHome').rename(
        columns={'xg':'home_xg_l','xgot':'home_xgot_l'}),on='match_id',how='left').merge(og_agg[og_agg.isHome == True].drop(columns='isHome').rename(
        columns={'own':'home_own'}),on='match_id',how='left')
    match_stats = match_stats.merge(shot_agg[shot_agg.isHome == False].drop(columns='isHome').rename(
        columns={'xg':'away_xg_l','xgot':'away_xgot_l'}),on='match_id',how='left').merge(og_agg[og_agg.isHome == False].drop(columns='isHome').rename(
        columns={'own':'away_own'}),on='match_id',how='left')
    match_stats.to_feather('data/match_stats.ftr')

def run_pipeline():
    results_list = create_results(league,season)
    completed_results = [x.replace(".ftr", "") for x in os.listdir(MATCH_DIR) if ".ftr" in x]
    results_process = list(set(results_list) - set(completed_results))
    print(f"Found {len(results_process)} new games.")
    for i in results_process:
        extract_match_summaries(i)

    completed_results = [x.replace(".ftr", "") for x in os.listdir(PLAYER_DIR) if ".ftr" in x]
    results_process = list(set(results_list) - set(completed_results))
    print(f"Found {len(results_process)} new games.")    
    for i in results_process:
        get_stats(i)

    completed_results = [x.replace(".ftr", "") for x in os.listdir(SHOTS_DIR) if ".ftr" in x]
    results_process = list(set(results_list) - set(completed_results))
    print(f"Found {len(results_process)} new games.")    
    for i in results_process:
        get_shots(i)

def transform_final_dataset():
    match_summaries = [x.replace(".ftr", "") for x in os.listdir(MATCH_DIR) if ".ftr" in x]
    match_summaries = list(map(int,match_summaries))
    print(len(match_summaries))
    match_stats = summarize_matches(league,match_summaries)
    match_stats.match_id = match_stats.match_id.astype('int')
    
    player_stats = [x.replace(".ftr", "") for x in os.listdir(PLAYER_DIR) if ".ftr" in x]
    player_stats = list(map(int,player_stats))
    print(len(player_stats))
    player_stats = summarize_players(player_stats)
    player_stats.match_id = player_stats.match_id.astype('int')
    
    shots = [x.replace(".ftr", "") for x in os.listdir(SHOTS_DIR) if ".ftr" in x]
    shots = list(map(int,shots))
    print(len(shots))
    shots = summarize_shots(shots)
    shots.match_id = shots.match_id.astype('int')

    finalize_matches(match_stats,shots)
    finalize_players(match_stats,player_stats)
    finalize_shots(match_stats,shots)

if __name__ == "__main__":
    run_pipeline() 
    transform_final_dataset()