import pandas as pd
import numpy as np
import os
from scipy.stats import skellam
from scipy.optimize import minimize_scalar

def calculate_parameters(results):
    results['hf'] = results.home_perf - results.away_perf
    hf = results.groupby('season').hf.mean().to_dict()

    results['tg'] = results.home_perf + results.away_perf
    tg = results.groupby('season').tg.mean().to_dict()

    return hf, tg 

def expected_points_calculator(row,n_sims=10000):
    home_og = int(row.home_own)
    away_og = int(row.away_own)
    home_shots = row.home_xg_l
    away_shots = row.away_xg_l

    home_array = np.concatenate([[1] * home_og,home_shots])
    away_array = np.concatenate([[1] * away_og,away_shots])

    home_probs = np.tile(home_array, n_sims)
    away_probs = np.tile(away_array, n_sims)

    home_random = np.random.rand(len(home_probs))
    away_random = np.random.rand(len(away_probs))

    home_goals = (home_random < home_probs).reshape(n_sims, -1).sum(axis=1)
    away_goals = (away_random < away_probs).reshape(n_sims, -1).sum(axis=1)

    home_points = np.where(home_goals > away_goals, 3,np.where(home_goals == away_goals, 1, 0))
    away_points = np.where(away_goals > home_goals, 3,np.where(home_goals == away_goals, 1, 0))

    return home_points.mean(), away_points.mean()

def calculate_standings(results):
    results[["home_xPts", "away_xPts"]] = results.apply(lambda row: expected_points_calculator(row, n_sims=10000),axis=1, result_type="expand")
    results['GP'] = 1
    temp_home = results[['season','home','away','GP','home_score','away_score','home_P','home_xg','away_xg','home_xPts']].reset_index(drop=True)
    temp_home.columns = temp_home.columns.str.replace('home','F').str.replace('away','A')
    temp_home['HR'] = 1

    temp_away = results[['season','home','away','GP','home_score','away_score','away_P','home_xg','away_xg','away_xPts']].reset_index(drop=True)
    temp_away.columns = temp_away.columns.str.replace('away','F').str.replace('home','A')
    temp_away['HR'] = -1

    results_t = pd.concat((temp_home,temp_away))
   
    standings = results_t.groupby(['season','F']).sum(numeric_only=True)
    standings['PPG'] = np.round(standings.F_P / standings.GP,2)
    standings['oRTG'] = np.round(standings.F_score / standings.GP * 0.3 + standings.F_xg / standings.GP * 0.7,2)
    standings['dRTG'] = np.round(standings.A_score / standings.GP * 0.3 + standings.A_xg / standings.GP * 0.7,2)
    standings['hRTG'] = np.round(standings.HR / standings.GP * (temp_home.F_xg.mean() - temp_away.F_xg.mean()) * -1,2)
    standings['nRTG'] = standings.oRTG - standings.dRTG + standings.hRTG
    standings = standings.sort_values('nRTG',ascending=False)

    return results, standings

def team_rating(xg,xga):
    win_rate = (1 - skellam.cdf(0,xg,xga)) + skellam.pmf(0,xg,xga) / 2
    return np.round(win_rate,6)

def define_dates_ratings(results):
    past_dates = results.game_date.drop_duplicates().sort_values().dt.date.values
    years = results.game_date.dt.year.drop_duplicates().sort_values()
    for year in years:
        past_dates = np.append(past_dates,pd.to_datetime(str(year)+'-01-01').date())
    past_dates = np.sort(past_dates)

    if past_dates[-1] == pd.to_datetime(str(years.iloc[-1])+'-01-01').date():
        past_dates = past_dates[:-1]
    
    return past_dates

def adjust_xg_xga(xg, xga, target_rating):
    delta_min = -xg  
    delta_max = xga     
    def loss(delta):
        return (team_rating(xg + delta, xga - delta) - target_rating)**2
    res = minimize_scalar(loss, bounds=(delta_min, delta_max), method='bounded')
    delta = res.x
    return (xg + delta)[0], (xga - delta)[0], target_rating[0]

def add_initial_season_ratings(team,team_ratings,season,team_initializations,transfer_vals):
    temp_season = int('20' + season)
    season_start_date = pd.to_datetime(str(temp_season) + '-01-01').date()
    season_transfer_date = pd.to_datetime(str(temp_season) + '-01-02').date()
    previous_season = str(int(season)-1)

    if team in team_ratings and previous_season in team_ratings[team]:
        season_data = team_ratings[team][previous_season]
        latest_date = sorted(season_data.keys())[-1]
        starting_rankings = season_data[latest_date]
    else:
        filtered = team_initializations[(team_initializations.team == team) &(team_initializations.season == season)]
                                        
        if filtered.empty:
            raise ValueError(f"No initialization found for {team} in {season}")
        else:
            starting_rankings = team_initializations[((team_initializations.team == team) & 
                                                    (team_initializations.season == season))][['ORtg','DRtg','WinRate']].iloc[0].values
    
    rating_adjustment = transfer_vals[(transfer_vals.team == team) & (transfer_vals.season == season)].Value.values[0]
    adjusted_rankings = np.round([starting_rankings[2] * 0.67 + rating_adjustment * 0.33],6)
    adjusted_rankings = list(adjust_xg_xga(starting_rankings[0],starting_rankings[1],adjusted_rankings))
    
    if team not in team_ratings:
        team_ratings[team] = {}
    if season not in team_ratings[team]:
        team_ratings[team][season] = {}

    team_ratings[team][season][season_start_date] = list(starting_rankings)
    team_ratings[team][season][season_transfer_date] = adjusted_rankings

def update_ratings(row,team_ratings,season,total_goals,home_field,update_rate):
    temp_date = row['game_date'].date()
    temp_home = row['home']
    temp_away = row['away']
    temp_home_perf = row['home_perf']
    temp_away_perf = row['away_perf']

    temp_home_rating = team_ratings[temp_home][season][sorted(team_ratings[temp_home][season].keys())[-1]]
    temp_away_rating = team_ratings[temp_away][season][sorted(team_ratings[temp_away][season].keys())[-1]]
    temp_home_off_perf = temp_home_perf / (total_goals/2 + home_field/2) / (temp_away_rating[1]/(total_goals/2))*total_goals/2
    temp_away_off_perf = temp_away_perf / (total_goals/2 - home_field/2) / (temp_home_rating[1]/(total_goals/2))*total_goals/2
    temp_home_def_perf = temp_away_perf / (total_goals/2 - home_field/2) / (temp_away_rating[0]/(total_goals/2))*total_goals/2
    temp_away_def_perf = temp_home_perf / (total_goals/2 + home_field/2) / (temp_home_rating[0]/(total_goals/2))*total_goals/2

    temp_home_rating_adj = [temp_home_rating[0] * (1 - update_rate) + temp_home_off_perf * update_rate,
                            temp_home_rating[1] * (1 - update_rate) + temp_home_def_perf * update_rate]
    temp_home_rating_adj.append(team_rating(temp_home_rating_adj[0],temp_home_rating_adj[1]))
    temp_away_rating_adj = [temp_away_rating[0] * (1 - update_rate) + temp_away_off_perf * update_rate,
                            temp_away_rating[1] * (1 - update_rate) + temp_away_def_perf * update_rate]
    temp_away_rating_adj.append(team_rating(temp_away_rating_adj[0],temp_away_rating_adj[1]))    

    team_ratings[temp_home][season][temp_date] = temp_home_rating_adj
    team_ratings[temp_away][season][temp_date] = temp_away_rating_adj

def normalize_ratings(team_ratings, target_date):
    rows = []
    for team, seasons in team_ratings.items():
        for season, dates in seasons.items():
            if target_date in dates:
                off, deff, rtg = dates[target_date]
                rows.append({'Team': team,'Season': season,'Date': target_date,'Off': off,'Def': deff,'Rtg': rtg})
    if not rows:
        print(f"No ratings found for {target_date}")
        return team_ratings
    df = pd.DataFrame(rows)

    mean_off = df['Off'].mean()
    mean_def = df['Def'].mean()
    target_mean = (mean_off + mean_def) / 2
    off_adjustment = mean_off - target_mean
    def_adjustment = mean_def - target_mean
    df['Off'] -= off_adjustment
    df['Def'] -= def_adjustment
    df['Rtg'] = team_rating(df.Off,df.Def)
    
    for _, row in df.iterrows():
        team = row['Team']
        season = row['Season']
        team_ratings[team][season][target_date] = [row['Off'], row['Def'], row['Rtg']]
    return team_ratings

def calculate_ratings(past_dates,transfer_vals,team_initializations,season_mapping,total_goals,home_field,results,update_rate):
    team_ratings = {}
    for date in past_dates:
        if date.month == 1 and date.day == 1:
            season = str(date.year)[2:]
            print(season)
            for team in transfer_vals[transfer_vals.season == season].team.values:
                add_initial_season_ratings(team,team_ratings,season,team_initializations,transfer_vals)    
            normalize_ratings(team_ratings, pd.to_datetime(f"{date.year}-01-01").date())
            normalize_ratings(team_ratings, pd.to_datetime(f"{date.year}-01-02").date())
        else:
            season = season_mapping[pd.to_datetime(date)]
            total_goals_season = total_goals[season]
            home_field_season = home_field[season]
            matches_temp = results[results.game_date.dt.date == date]

            for idx, row in matches_temp.iterrows():
                update_ratings(row,team_ratings,season,total_goals_season,home_field_season,update_rate)
    return team_ratings

def define_dates_sims(results):
    past_dates = results.game_date.drop_duplicates().sort_values().dt.date.values
    years = results.game_date.dt.year.drop_duplicates().sort_values()
    for year in years:
        past_dates = np.append(past_dates,pd.to_datetime(str(year)+'-01-01').date())
        past_dates = np.append(past_dates,pd.to_datetime(str(year)+'-01-02').date())
    past_dates = np.sort(past_dates)

    if past_dates[-2] == pd.to_datetime(str(years.iloc[-1])+'-01-01').date():
        past_dates = past_dates[:-2]
    return past_dates

def find_matches(date,season,temp_matches):
    temp_matches = temp_matches.copy()
    temp_matches['GP'] = 1
    temp_matches = temp_matches[temp_matches.season == season]
    temp_results = temp_matches[temp_matches.game_date.dt.date <= date][['game_date','home','away','home_score','away_score','GP']]
    temp_schedule = temp_matches[temp_matches.game_date.dt.date > date][['game_date','home','away','GP']]
    return temp_schedule, temp_results

def find_ratings_all(ratings, season, date):
    result = {}
    for team, team_data in ratings.items():
        if season not in team_data:
            continue
        season_data = team_data[season]
        valid_dates = [d for d in season_data if d <= date]
        if not valid_dates:
            continue
        latest_date = max(valid_dates)
        result.setdefault(team, {})[season] = {latest_date: season_data[latest_date]}
    return result

def find_ratings(ratings,team,season,date):
    season_data = ratings[team][season]
    valid_dates = [d for d in season_data.keys() if d <= date]
    latest_date = max(valid_dates)
    result = season_data[latest_date]
    return result

def prepare_rating_arrays(temp_ratings, temp_schedule, temp_results, season, date):
    teams = pd.concat([temp_schedule['home'],temp_schedule['away'],temp_results['home'],temp_results['away']]).unique()
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    n_teams = len(teams)
    ratings_array = np.zeros((n_teams, 2))
    for team in teams:
        rating = find_ratings(temp_ratings, team, season, date)
        ratings_array[team_to_idx[team]] = rating[:2]
    return ratings_array, team_to_idx, teams

def simulate_season_vectorized(schedule_home_idx, schedule_away_idx, initial_ratings,temp_goals, temp_home_field, update_rate):
    n_matches = len(schedule_home_idx)
    ratings = initial_ratings.copy()
    
    home_goals = np.zeros(n_matches, dtype=int)
    away_goals = np.zeros(n_matches, dtype=int)
    
    for i in range(n_matches):
        home_idx = schedule_home_idx[i]
        away_idx = schedule_away_idx[i]
        home_off, home_def = ratings[home_idx, 0], ratings[home_idx, 1]
        away_off, away_def = ratings[away_idx, 0], ratings[away_idx, 1]
        
        temp_home_xg = (home_off / (temp_goals/2)) * (away_def / (temp_goals/2)) * (temp_goals/2 + temp_home_field/2)
        temp_away_xg = (away_off / (temp_goals/2)) * (home_def / (temp_goals/2)) * (temp_goals/2 - temp_home_field/2)
        h_g = np.random.poisson(temp_home_xg)
        a_g = np.random.poisson(temp_away_xg)
        home_goals[i] = h_g
        away_goals[i] = a_g
        home_perf = temp_home_xg * 0.7 + h_g * 0.3
        away_perf = temp_away_xg * 0.7 + a_g * 0.3
        
        home_off_perf = home_perf / (temp_goals/2 + temp_home_field/2) / (away_def/(temp_goals/2)) * (temp_goals/2)
        away_off_perf = away_perf / (temp_goals/2 - temp_home_field/2) / (home_def/(temp_goals/2)) * (temp_goals/2)
        home_def_perf = away_perf / (temp_goals/2 - temp_home_field/2) / (away_off/(temp_goals/2)) * (temp_goals/2)
        away_def_perf = home_perf / (temp_goals/2 + temp_home_field/2) / (home_off/(temp_goals/2)) * (temp_goals/2)
        ratings[home_idx, 0] = home_off * (1 - update_rate) + home_off_perf * update_rate
        ratings[home_idx, 1] = home_def * (1 - update_rate) + home_def_perf * update_rate
        ratings[away_idx, 0] = away_off * (1 - update_rate) + away_off_perf * update_rate
        ratings[away_idx, 1] = away_def * (1 - update_rate) + away_def_perf * update_rate
    
    return home_goals, away_goals

def fast_table_from_goals(home_goals, away_goals, home_teams, away_teams,results_summary, team_to_idx):
    n_teams = len(team_to_idx)
    points = np.zeros(n_teams)
    goals_for = np.zeros(n_teams)
    goals_against = np.zeros(n_teams)
    
    home_wins = (home_goals > away_goals).astype(int)
    away_wins = (away_goals > home_goals).astype(int)
    draws = (home_goals == away_goals).astype(int)
    
    for i in range(len(home_goals)):
        h_idx = home_teams[i]
        a_idx = away_teams[i]
        
        points[h_idx] += home_wins[i] * 3 + draws[i]
        points[a_idx] += away_wins[i] * 3 + draws[i]
        
        goals_for[h_idx] += home_goals[i]
        goals_for[a_idx] += away_goals[i]
        goals_against[h_idx] += away_goals[i]
        goals_against[a_idx] += home_goals[i]
    
    if results_summary is not None:
        for team, idx in team_to_idx.items():
            if team in results_summary.index:
                points[idx] += results_summary.loc[team, 'Points']
                goals_for[idx] += results_summary.loc[team, 'F']
                goals_against[idx] += results_summary.loc[team, 'A']
    
    goal_diff = goals_for - goals_against
    
    ranks = np.zeros(n_teams, dtype=int)
    for rank, idx in enumerate(np.lexsort((-goals_for, -goal_diff, -points)), 1):
        ranks[idx] = rank
    
    return ranks, points, goals_for, goals_against, goal_diff

def summarize_matches(results):
    summary = pd.concat((
        results[['home','home_score','away_score']].rename(columns={'home':'Team','home_score':'F','away_score':'A'}),
        results[['away','away_score','home_score']].rename(columns={'away':'Team','away_score':'F','home_score':'A'})))

    summary['Points'] = ((summary.F > summary.A).astype('int') * 3 + (summary.F == summary.A).astype('int'))
    summary['Goal_D'] = summary.F - summary.A
    summary = summary.groupby('Team').sum().sort_values(['Points','Goal_D','F'],ascending=[False,False,False])
    summary['rank'] = summary[['Points','Goal_D','F']].apply(tuple,axis=1).rank(method='dense',ascending=False).astype('int')
    return summary

def simulate_individual_matches(temp_schedule, ratings_array, team_to_idx,temp_goals, temp_home_field, n_sims):
    n_matches = len(temp_schedule)
    
    home_exp = np.zeros(n_matches)
    away_exp = np.zeros(n_matches)
    home_wins = np.zeros(n_matches)
    away_wins = np.zeros(n_matches)
    draws = np.zeros(n_matches)

    all_match_stats = []
    for i, (idx, row) in enumerate(temp_schedule.iterrows()):
        temp_date = row['game_date']
        temp_home = row['home']
        temp_away = row['away']
        
        home_idx = team_to_idx[temp_home]
        away_idx = team_to_idx[temp_away]
        home_off, home_def = ratings_array[home_idx, 0], ratings_array[home_idx, 1]
        away_off, away_def = ratings_array[away_idx, 0], ratings_array[away_idx, 1]   
        temp_home_exp = (home_off / (temp_goals/2)) * (away_def / (temp_goals/2)) * (temp_goals/2 + temp_home_field/2)
        temp_away_exp = (away_off / (temp_goals/2)) * (home_def / (temp_goals/2)) * (temp_goals/2 - temp_home_field/2)
        temp_home_arr = np.random.poisson(temp_home_exp, n_sims)
        temp_away_arr = np.random.poisson(temp_away_exp, n_sims)
        temp_home_win = np.sum(temp_home_arr > temp_away_arr) / n_sims
        temp_away_win = np.sum(temp_home_arr < temp_away_arr) / n_sims
        temp_draw = np.sum(temp_home_arr == temp_away_arr) / n_sims
        
        home_goals_unique, home_goals_counts = np.unique(temp_home_arr, return_counts=True)
        away_goals_unique, away_goals_counts = np.unique(temp_away_arr, return_counts=True)
        temp_home_dist = dict(zip(home_goals_unique, home_goals_counts / n_sims))
        temp_away_dist = dict(zip(away_goals_unique, away_goals_counts / n_sims))
        
        temp_stats = pd.DataFrame({'game_date': [temp_date],'Home': [temp_home],'Away': [temp_away],'h_exp': [temp_home_exp],'a_exp': [temp_away_exp],
                                   'h_win': [temp_home_win],'d_win': [temp_draw],'a_win': [temp_away_win]})
        
        for goal, prob in temp_home_dist.items():
            temp_stats[f'h_{goal}'] = prob
        
        for goal, prob in temp_away_dist.items():
            temp_stats[f'a_{goal}'] = prob
        all_match_stats.append(temp_stats)
        
    return pd.concat(all_match_stats, ignore_index=True)

def simulate_matchups(sim_dates,matches,team_ratings,total_goals,home_field,n_sims):
    for date in sim_dates:
        season = str(date.year)[2:]
        
        temp_schedule, temp_results = find_matches(date, season, matches)
        temp_ratings = find_ratings_all(team_ratings, season, date)
        temp_goals = total_goals[season]
        temp_home_field = home_field[season]
        
        if len(temp_schedule) != 0:
            ratings_array, team_to_idx, teams = prepare_rating_arrays(temp_ratings, temp_schedule, temp_results, season, date)
            match_sims = simulate_individual_matches(temp_schedule, ratings_array, team_to_idx,temp_goals, temp_home_field, n_sims)
            match_sims.insert(0, 'Sim_Date', date)
            
            match_sims.to_feather(f'data/Sim_States/{date}_matches.ftr')

def simulate_season(sim_dates,matches,total_goals,home_field,n_sims,update_rate,team_ratings,conferences):
    for date in sim_dates:
        season = str(date.year)[2:]

        temp_schedule, temp_results = find_matches(date, season, matches)
        temp_ratings = find_ratings_all(team_ratings, season, date)
        temp_goals = total_goals[season]
        temp_home_field = home_field[season]
        
        if len(temp_schedule) != 0:
            ratings_array, team_to_idx, teams = prepare_rating_arrays(temp_ratings, temp_schedule, temp_results, season, date)
            schedule_home_idx = temp_schedule['home'].map(team_to_idx).values
            schedule_away_idx = temp_schedule['away'].map(team_to_idx).values
            
            results_summary = summarize_matches(temp_results) if len(temp_results) > 0 else None
            all_ranks = np.zeros((n_sims, len(teams)), dtype=int)
            all_conf_ranks = np.zeros((n_sims, len(teams)), dtype=int)
            all_points = np.zeros((n_sims, len(teams)))
            all_gf = np.zeros((n_sims, len(teams)))
            all_ga = np.zeros((n_sims, len(teams)))
            all_gd = np.zeros((n_sims, len(teams)))

            team_conferences = np.array([conferences.get(team) for team in teams])
            unique_confs = np.unique(team_conferences)
            conf_masks = {conf: np.where(team_conferences == conf)[0] for conf in unique_confs}
            
            for sim in range(n_sims):
                home_goals, away_goals = simulate_season_vectorized(schedule_home_idx, schedule_away_idx, ratings_array,temp_goals,temp_home_field, update_rate)
                ranks, points, gf, ga, gd = fast_table_from_goals(home_goals,away_goals,schedule_home_idx, schedule_away_idx,results_summary,team_to_idx)
                all_ranks[sim] = ranks
                all_points[sim] = points
                all_gf[sim] = gf
                all_ga[sim] = ga
                all_gd[sim] = gd

                conf_ranks = np.zeros(len(teams), dtype=int)
                for conf_indices in conf_masks.values():
                    conf_points = points[conf_indices]
                    conf_gd = gd[conf_indices]
                    order = np.lexsort(((-conf_gd), (-conf_points)))
                    for conf_rank, local_idx in enumerate(order, start=1):
                        conf_ranks[conf_indices[local_idx]] = conf_rank
                all_conf_ranks[sim] = conf_ranks
            
            sim_results = pd.DataFrame({'Points': all_points.mean(axis=0),'F': all_gf.mean(axis=0),'A': all_ga.mean(axis=0),'Goal_D': all_gd.mean(axis=0),
                                        'rank': all_ranks.mean(axis=0),'conf_rank': all_conf_ranks.mean(axis=0)}, index=teams)
            for rank in range(1, len(teams) + 1):
                sim_results[str(rank)] = (all_ranks == rank).mean(axis=0)
            max_conf_size = max(len(v) for v in conf_masks.values())
            for rank in range(1, max_conf_size + 1):
                sim_results[f'conf_{rank}'] = (all_conf_ranks == rank).mean(axis=0)

        else:
            results_summary = summarize_matches(temp_results)
            sim_results = results_summary.copy()
            n_teams = len(results_summary)
            for rank in range(1, n_teams + 1):
                sim_results[str(rank)] = 0.0
            for team in results_summary.index:
                actual_rank = results_summary.loc[team, 'rank']
                sim_results.loc[team, str(actual_rank)] = 1.0
                
        sim_results.to_feather(f'data/Sim_States/{date}.ftr')

def clean_team_ratings(team_ratings):
    rows = []
    for team, seasons in team_ratings.items():
        for season, dates in seasons.items():
            for date, values in dates.items():
                rows.append({
                    'Team': team,
                    'Season': season,
                    'Date': date,
                    'A': values[0],
                    'B': values[1],
                    'C': values[2]
                })
    clean_team_ratings = pd.DataFrame(rows)
    return clean_team_ratings

def run_main(update_rate = 2/38,n_sims = 10000):
    schedule = pd.read_csv('data/Schedule.csv')
    schedule.game_date = pd.to_datetime(pd.to_datetime(schedule.game_date,unit='s').dt.date)
    schedule.season = schedule.season.astype('str')

    results = pd.read_feather('data/match_stats.ftr')
    results.game_date = pd.to_datetime(pd.to_datetime(results.game_date,unit='s').dt.date)
    results.season = results.season.astype('str')
    results = results.sort_values(['season','game_date'])
    results['home_P'] = (results.home_score > results.away_score).astype('int') * 3 + (results.home_score == results.away_score).astype('int')
    results['away_P'] = (results.home_score < results.away_score).astype('int') * 3 + (results.home_score == results.away_score).astype('int')
    results['home_perf'] = results.home_xg * 0.7 + results.home_score * 0.3
    results.home_perf = results.home_perf.fillna(results.home_score)
    results['away_perf'] = results.away_xg * 0.7 + results.away_score * 0.3
    results.away_perf = results.away_perf.fillna(results.away_score)

    color_map = results[['home','home_primary','home_secondary','home_text']].drop_duplicates().sort_values('home').set_index('home')
    color_map.to_feather('data/color_map.ftr')

    initial_ratings = pd.read_csv('data/Initializations.txt')
    initial_ratings.season = initial_ratings.season.astype('int').astype('str')
    initial_ratings['WinRate'] = initial_ratings.apply(lambda row: team_rating(row['ORtg'], row['DRtg']), axis=1)
    conferences = initial_ratings.set_index('team').conference.to_dict()

    ##'https://www.transfermarkt.com/major-league-soccer/marktwerteverein/wettbewerb/MLS1/plus/1?stichtag=2023-08-01'
    transfer_values = pd.read_csv('data/TransferMarkt.txt')
    transfer_values.season = transfer_values.season.astype('str')
    transfer_values['mean'] = transfer_values.groupby('season').Value.transform('mean')
    transfer_values['std'] = transfer_values.groupby('season').Value.transform('std')
    transfer_values.Value = (transfer_values.Value - transfer_values['mean'])/transfer_values['std']
    transfer_values.Value = (transfer_values.Value * 0.3 + 1.5)/3

    hf, tg = calculate_parameters(results[results.type == 'regular'].reset_index(drop=True))
    results, standings = calculate_standings(results[results.type == 'regular'].reset_index(drop=True))

    matches = pd.concat((results,schedule)).sort_values(['season','game_date'])
    season_mapping = matches[['season','game_date']].drop_duplicates()
    season_mapping = season_mapping.set_index('game_date').to_dict()['season']
    past_dates = define_dates_ratings(results)
    team_ratings = calculate_ratings(past_dates,transfer_values,initial_ratings,season_mapping,tg,hf,results,update_rate)

    past_dates = define_dates_sims(results)
    prev_sims = os.listdir('data/Sim_States')
    prev_sims = list(filter(lambda k: '_matches' not in k, prev_sims))
    prev_sims = list(filter(lambda k: '.ftr' in k, prev_sims))
    prev_sims = [s.replace('.ftr', '') for s in prev_sims]
    prev_sims = pd.to_datetime(prev_sims).date
    sim_dates = sorted(list(set(past_dates) - set(prev_sims)))
    print(len(past_dates),len(prev_sims),len(sim_dates))
    simulate_season(sim_dates,matches[matches.type == 'regular'],tg,hf,n_sims,update_rate,team_ratings,conferences)
    simulate_matchups(sim_dates,matches,team_ratings,tg,hf,n_sims)
    matches.reset_index(drop=True).drop(columns = ['Unnamed: 0','league']).to_feather('data/matches.ftr')
    clean_team_ratings(team_ratings).to_feather('data/team_ratings.ftr')
    standings.reset_index().to_feather('data/standings.ftr')

if __name__ == "__main__":
    run_main()