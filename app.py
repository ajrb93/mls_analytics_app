import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import base64
from io import BytesIO
import numpy as np

# --- 1. CONFIG & COMPACT STYLING ---
st.set_page_config(layout="wide", page_title="Major League Soccer")

# CUSTOM CSS: Shrinks headers, table padding, and overall container gaps
st.markdown("""
    <style>
    /* Page margins */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Tab labels */
    button[data-baseweb="tab"] {
        font-size: 14px !important;
    }
    button[data-baseweb="tab"] div {
        font-size: 14px !important;
    }

    /* Expander headers */
    div[data-testid="stExpander"] div[role="button"] p { 
        font-size: 12px !important; 
        font-weight: bold !important; 
    }
    </style>
""", unsafe_allow_html=True)

# Load All Data
def credible_range_str(row, level=0.9):
    probs = row.sort_values(ascending=False)
    cumsum = probs.cumsum()
    included = probs.index[cumsum <= level]
    if len(included) < len(probs):
        included = included.append(pd.Index([cumsum.index[len(included)]]))
    nums = [int(float(p)) for p in included]
    lo, hi = min(nums), max(nums)
    return f"{lo}" if lo == hi else f"{lo} to {hi}"

def load_standings_sims():
    files = os.listdir('data/Sim_States/')
    files = list(filter(lambda k: '.ftr' in k, files))

    standings_files = list(filter(lambda k: '_matches' not in k, files))
    match_files = list(filter(lambda k: '_matches' in k, files))

    standings_sims = []
    for file in standings_files:
        temp = pd.read_feather('data/Sim_States/'+file)
        date = file.replace('.ftr','')
        temp['Sim_Date'] = date
        standings_sims.append(temp)
    standings_sims = pd.concat((standings_sims)).reset_index()
    standings_sims.Sim_Date = pd.to_datetime(standings_sims.Sim_Date).dt.date
    standings_sims = standings_sims.fillna(0)
    standings_sims['Champ'] = standings_sims['1']
    standings_sims['HomeField'] = standings_sims[['conf_1','conf_2','conf_3','conf_4']].sum(axis=1)
    standings_sims['Playoffs'] = standings_sims[['conf_1','conf_2','conf_3','conf_4','conf_5','conf_6','conf_7','conf_8','conf_9']].sum(axis=1)

    standings_sims['range'] = standings_sims[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25',
                                               '26','27','28','29','30']].apply(credible_range_str, axis=1)
    standings_sims['season'] = (pd.to_datetime(standings_sims.Sim_Date).dt.year).astype('str').str[2:]
    
    match_sims = []
    for file in match_files:
        temp = pd.read_feather('data/Sim_States/'+file)
        match_sims.append(temp)
    match_sims = pd.concat((match_sims))

    return standings_sims, match_sims

def create_standings_file(standings,standings_sims,team_ratings,season,max_date,min_date):
    temp = standings[standings.season == season][['season','F','F_score','A_score','F_P','F_xg','A_xg','F_xPts','oRTG','dRTG','nRTG']].reset_index(drop=True)
    temp['GD'] = temp.F_score - temp.A_score
    temp['xGD'] = temp.F_xg - temp.A_xg
    temp_sim = standings_sims[standings_sims.Sim_Date == max_date].set_index('index')[['Points','Champ','CL','Rel','range']]
    temp_sim2 = standings_sims[standings_sims.Sim_Date == min_date].set_index('index')[['Points','Champ','CL','Rel']]
    temp_sim2 = temp_sim - temp_sim2
    temp_sim3 = team_ratings[team_ratings.Date == max_date].set_index('Team')[['Date','A','B','C']]
    temp_sim4 = team_ratings[team_ratings.Date == min_date].set_index('Team')[['A','B','C']]
    temp_sim4 = temp_sim3 - temp_sim4
    temp = temp.merge(temp_sim.reset_index(),left_on='F',right_on='index').merge(temp_sim2.reset_index(),left_on='F',right_on='index',suffixes=['','_c']).merge(
        temp_sim3.reset_index(),left_on='F',right_on='Team').merge(temp_sim4.reset_index(),left_on='F',right_on='Team',suffixes=['','_c'])
    temp = temp[['season','Team','C','C_c','A','A_c','B','B_c','nRTG','oRTG','dRTG','Points','Points_c','F_P','F_xPts','GD','xGD','Champ','Champ_c','CL','CL_c','Rel',
                 'Rel_c','range']].rename(
                     columns={'A':'oPRE','A_c':'oPREΔ','B':'dPRE','B_c':'dPREΔ','F_P':'P','F_xPts':'xPts','Points':'Proj','Points_c':'ProjΔ','C':'nPRE','C_c':'nPREΔ',
                               'Champ':'Win','Champ_c':'WinΔ','CL_c':'CLΔ','Rel_c':'RelΔ'})
    return temp

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3],16) for i in range(0,lv,lv//3))

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

def mean_color(color1,color2):
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    
    avg = lambda x,y: round((x+y)/2)
    new_rgb = ()
    for i in range(len(rgb1)):
        new_rgb += (avg(rgb1[i],rgb2[i]),)
    
    return '#' + rgb_to_hex(new_rgb)

#colormap
colors = [(0.75,0,0),(1,1,1),(0,0.75,0)]
colors_r = [(0,0.75,0),(1,1,1),(0.75,0,0),]
n_bins = 100
cmap = mcolors.LinearSegmentedColormap.from_list('redwhitegreen',colors,N=n_bins)
cmap_r = mcolors.LinearSegmentedColormap.from_list('redwhitegreen_r',colors_r,N=n_bins)
norm_o = mcolors.TwoSlopeNorm(vmin=0,vcenter=1.3,vmax=2.6)
norm_r = mcolors.TwoSlopeNorm(vmin=0,vcenter=1,vmax=3)
norm_p = mcolors.TwoSlopeNorm(vmin=0,vcenter=1.5,vmax=3)
norm_w = mcolors.TwoSlopeNorm(vmin=0,vcenter=1/3,vmax=1)
norm_perf = mcolors.TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)

def plot_standings_table(standings_df):
    fig, ax = plt.subplots(figsize=(12,12/2.33333333))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # --- HEADERS ---
    ax.annotate('Team',       (0.01,  0.97), va='center', ha='left',   size=10, weight='bold')
    ax.annotate('Skill',      (1.65/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Off',        (2.65/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Def',        (3.65/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Performance',(4.65/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Proj',       (5.5/10,  0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Points',     (6.15/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('GD',         (6.7/10,  0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Champ',      (7.45/10, 0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('CL',         (8.3/10,  0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Rele.',      (9.1/10,  0.97), va='center', ha='center', size=10, weight='bold')
    ax.annotate('Range',      (9.75/10, 0.97), va='center', ha='center', size=10, weight='bold')

    # --- VERTICAL DIVIDERS ---
    for x in [1.15, 2.15, 3.15, 4.15, 5.15, 5.85, 6.45, 7.05, 7.85, 8.65, 9.50]:
        ax.axvline(x/10, color='black', linewidth=0.5)

    # --- ROWS ---
    n_teams = len(standings_df)
    top = 0.93
    bottom_margin = 0.01
    total_height = top - bottom_margin
    space = total_height / n_teams
    i_loc = top - space / 2

    ax.vlines(4.483/10, bottom_margin, top, color='black', linewidth=0.3, linestyle='--')
    ax.vlines(4.816/10, bottom_margin, top, color='black', linewidth=0.3, linestyle='--')

    for _, row in standings_df.iterrows():
        team = row['Team']

        # Team name
        ax.annotate(team, (0.01, i_loc), va='center', ha='left', size=9, fontweight='bold',color=team_colors[team]['home_secondary'])
        ax.add_patch(Rectangle((0,i_loc+space/2),1.15/10,-space,facecolor=team_colors[team]['home_primary']))
        ax.add_patch(Rectangle((1.15/10, i_loc - space/2), 1, space, facecolor = mean_color(mean_color(team_colors[team]['home_primary'],'#FFFFFF'),'#FFFFFF')))

        # Skill (nPRE)
        ax.annotate(f"{row['nPRE']:.0%}", (1.4/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['nPREΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['nPREΔ'] > 0 else ''}{row['nPREΔ']:.0%})", (1.9/10, i_loc), va='center', ha='center', size=9, color=delta_color)
        ax.add_patch(Rectangle((1.15/10, i_loc - space/2), 0.5/10, space,facecolor=cmap(row['nPRE'])))

        # Offensive (oPRE)
        ax.annotate(f"{row['oPRE']:.2f}", (2.4/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['oPREΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['oPREΔ'] > 0 else ''}{row['oPREΔ']:.0%})", (2.9/10, i_loc), va='center', ha='center', size=9, color=delta_color)
        ax.add_patch(Rectangle((2.15/10, i_loc - space/2), 0.5/10, space,facecolor=cmap(norm_o(row['oPRE']))))

        # Defensive (dPRE)
        ax.annotate(f"{row['dPRE']:.2f}", (3.4/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['dPREΔ'] < 0 else 'darkred'
        ax.annotate(f"({'+' if row['dPREΔ'] < 0 else ''}{row['dPREΔ']*-1:.0%})", (3.9/10, i_loc), va='center', ha='center', size=9, color=delta_color)
        ax.add_patch(Rectangle((3.15/10, i_loc - space/2), 0.5/10, space,facecolor=cmap(1 - norm_o(row['dPRE']))))

        # Performance (nRTG, oRTG, dRTG)
        ax.add_patch(Rectangle((4.15/10, i_loc - space/2), (1/3)/10, space,facecolor=cmap(norm_perf(row['nRTG']))))
        ax.add_patch(Rectangle((4.483/10, i_loc - space/2), (1/3)/10, space,facecolor=cmap(norm_o(row['oRTG']))))
        ax.add_patch(Rectangle((4.816/10, i_loc - space/2), (1/3)/10, space,facecolor=cmap(1 - norm_o(row['dRTG']))))
        ax.annotate(f"{row['nRTG']:.2f}", (4.32/10, i_loc), va='center', ha='center', size=9)
        ax.annotate(f"{row['oRTG']:.2f}", (4.65/10, i_loc), va='center', ha='center', size=9)
        ax.annotate(f"{row['dRTG']:.2f}", (4.97/10, i_loc), va='center', ha='center', size=9)


        # Proj + ProjΔ
        ax.annotate(f"{row['Proj']:.0f}", (5.3/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['ProjΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['ProjΔ'] > 0 else ''}{row['ProjΔ']:.0f})", (5.65/10, i_loc), va='center', ha='center', size=9, color=delta_color)

        # Points + xPts
        ax.annotate(f"{int(row['P'])}", (6.0/10, i_loc), va='center', ha='center', size=9)
        ax.annotate(f"{row['xPts']:.1f}", (6.25/10, i_loc), va='center', ha='center', size=9)

        # GD + xGD
        ax.annotate(f"{int(row['GD'])}", (6.6/10, i_loc), va='center', ha='center', size=9)
        ax.annotate(f"{row['xGD']:.1f}", (6.85/10, i_loc), va='center', ha='center', size=9)

        # Champ + WinΔ
        ax.annotate(f"{row['Win']:.0%}", (7.25/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['WinΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['WinΔ'] > 0 else ''}{row['WinΔ']:.0%})", (7.65/10, i_loc), va='center', ha='center', size=9, color=delta_color)

        # CL + CLΔ
        ax.annotate(f"{row['CL']:.0%}", (8.05/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['CLΔ'] > 0 else 'darkred'
        ax.annotate(f"({'+' if row['CLΔ'] > 0 else ''}{row['CLΔ']:.0%})", (8.45/10, i_loc), va='center', ha='center', size=9, color=delta_color)

        # Rel + RelΔ
        ax.annotate(f"{row['Rel']:.0%}", (8.85/10, i_loc), va='center', ha='center', size=9)
        delta_color = 'darkgreen' if row['RelΔ'] < 0 else 'darkred'
        ax.annotate(f"({'+' if row['RelΔ'] < 0 else ''}{row['RelΔ']*-1:.0%})", (9.25/10, i_loc), va='center', ha='center', size=9, color=delta_color)

        # Range
        ax.annotate(row['range'], (9.75/10, i_loc), va='center', ha='center', size=9)

        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.5)

        i_loc -= space

    # Top border
    ax.axhline(0.935, color='black', linewidth=0.5)

    plt.tight_layout()
    return fig

def plot_ratings_scatter(standings_df, team_colors):
    fig = go.Figure()

    off_mean = standings_df['oPRE'].mean()
    def_mean = standings_df['dPRE'].mean()

    # Diagonal reference lines (equivalent to your matplotlib lines)
    for offset in [-2/3, -1/3, 0, 1/3, 2/3]:
        x_start = def_mean * 1.5
        x_end = def_mean * 0.5
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[x_start + offset * off_mean, x_end + offset * off_mean],
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Arrows showing movement
    for _, row in standings_df.iterrows():
        team = row['Team']
        primary = team_colors[team]['home_primary']
        x_start = row['dPRE'] - row['dPREΔ']
        y_start = row['oPRE'] - row['oPREΔ']
        x_end = row['dPRE']
        y_end = row['oPRE']

        # Arrow line
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[y_start, y_end],
            mode='lines',
            line=dict(color=primary, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Scatter points at current position
    fig.add_trace(go.Scatter(
        x=standings_df['dPRE'],
        y=standings_df['oPRE'],
        mode='markers',
        marker=dict(
            color=[team_colors[t]['home_primary'] for t in standings_df['Team']],
            size=12,
            line=dict(
                color=[team_colors[t]['home_secondary'] for t in standings_df['Team']],
                width=2
            )
        ),
        text=standings_df['Team'],
        hovertemplate='<b>%{text}</b><br>Off: %{y:.2f}<br>Def: %{x:.2f}<extra></extra>',
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(
            range=[def_mean * 1.5, def_mean * 0.5],  # inverted — lower def is better
            title='Defensive Rating',
            showgrid=False
        ),
        yaxis=dict(
            range=[off_mean * 0.5, off_mean * 1.5],
            title='Offensive Rating',
            showgrid=False,
            scaleanchor='x',
            scaleratio=1
        ),
        plot_bgcolor='gainsboro',
        margin=dict(l=20, r=20, t=20, b=20),
        height = 400,
        width = 400
    )

    return fig

def plot_position_heatmap(standings_sims, standings_df, selected_end_date, team_colors):
    # Get position probabilities for selected date, ordered by current standings
    sim_data = standings_sims[standings_sims.Sim_Date == selected_end_date].set_index('index')
    position_cols = [str(i) for i in range(1, 21)]
    
    # Order teams by current points (same order as standings table)
    teams_ordered = standings_df['Team'].tolist()
    
    heatmap_data = sim_data.loc[teams_ordered, position_cols].astype(float)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=list(range(1, 21)),
        y=teams_ordered,
        colorscale='RdYlGn',
        showscale=False,
        hovertemplate='<b>%{y}</b><br>Position %{x}: %{z:.0%}<extra></extra>',
        zmin=0,
        zmax=1,
        text=[[f"{val*100:.0f}" if val >= 0.005 else "" for val in row] for row in heatmap_data.values],
        texttemplate="%{text}",
        textfont=dict(size=9)
    ))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(
            autorange='reversed',  # keep standings order top to bottom
            tickfont=dict(size=10)
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        width=820,  # 400px plot + room for team names on left
        height=400
    )

    return fig

def create_matches_df(match_sims,matches,team_ratings,selected_season,selected_end_date):
    match_sims = match_sims[match_sims.Sim_Date <= selected_end_date].sort_values('Sim_Date').groupby(['game_date','Home','Away']).tail(1)
    match_sims = match_sims[['Sim_Date','game_date','Home','Away','h_exp','a_exp','h_win','d_win','a_win']]
    matches = matches[matches.season == selected_season]
    matches = matches[['game_date','home','away','home_score','away_score','home_xg','away_xg','home_P','away_P','home_perf','away_perf','home_xPts','away_xPts']]
    matches.loc[matches.game_date.dt.date > selected_end_date,('home_score','away_score','home_xg','away_xg','home_P','away_P','home_perf','away_perf',
                                                       'home_xPts','away_xPts')] = np.nan
    matches = matches.merge(match_sims,left_on=['game_date','home','away'],right_on=['game_date','Home','Away']).drop(columns=['Home','Away'])

    plot_df = matches.merge(
        team_ratings.drop(columns=['Season','A','B']), left_on=['Sim_Date','home'], right_on=['Date','Team']).merge(
        team_ratings.drop(columns=['Season','A','B']), left_on=['Sim_Date','away'], right_on=['Date','Team'], suffixes=['_H','_A']).drop(
        columns=['Sim_Date','Date_H','Date_A'])
    plot_df['Pre_Pts_H'] = plot_df.h_win * 3 + plot_df.d_win
    plot_df['Pre_Pts_A'] = plot_df.a_win * 3 + plot_df.d_win
    return plot_df.sort_values('game_date').reset_index(drop=True)

def create_results_figure(plot_df):
    results = plot_df[~plot_df.home_score.isna()].reset_index(drop=True).sort_values('game_date',ascending=False)
    results[' '] = ''
    results['score'] = results.home_score.astype('int').astype('str') + ' - ' + results.away_score.astype('int').astype('str')
    results = results[['game_date','home','Pre_Pts_H','score','Pre_Pts_A','away',' ','home_xPts','away_xPts',' ','home_xg','away_xg','home_score','away_score']].rename(
        columns={'game_date':'Date','home':'Home','Pre_Pts_H':'H_F','Pre_Pts_A':'A_F','away':'Away','home_xPts':'Per_H','away_xPts':'Per_A',
                 'home_xg':'xG_H','away_xg':'xG_A'})

    fig_height = max(4, len(results) * 0.2)
    fig, ax = plt.subplots(figsize=(7,fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Column x positions
    col_x = {
        'Date':   0.01,
        'Home':   0.12,
        '':    0.37,
        'Away':   0.42,
        '1':0.67,
        'HExp':0.68,
        'HPer':  0.73,
        '3':0.78,
        'AExp':0.79,
        'APer':  0.84,
        '2':0.89,
        'HxG':   0.90,
        'AxG':   0.95
    }

    # Headers
    header_y = (len(results)+0.5)/(len(results)+1)
    for col, x in col_x.items():
        if (col == '1') | (col == '2') | (col == '3'):
            pass
        else:
            ha = 'left'
            ax.annotate(col, (x, header_y), va='center', ha=ha, size=7, weight='bold')

    # Row layout
    top = len(results)/(len(results)+1)
    ax.axhline(top, color='black', linewidth=0.8)
    bottom_margin = 1/(len(results)+1)/10
    total_height = top - bottom_margin
    space = total_height / max(4, len(results))
    i_loc = top - space / 2

    # Vertical dividers (between groups)
    for x in col_x.values():
        ax.vlines(x-0.005, bottom_margin, top, color='black', linewidth=0.5)

    for _, row in results.iterrows():
        home_primary = team_colors[row['Home']]['home_primary']
        home_text = team_colors[row['Home']]['home_secondary']
        away_primary = team_colors[row['Away']]['home_primary']
        away_text = team_colors[row['Away']]['home_secondary']

        # Home team background
        ax.add_patch(Rectangle((0.12-0.005, i_loc - space/2), 0.25, space, facecolor=home_primary))
        # Away team background
        ax.add_patch(Rectangle((0.42-0.005, i_loc - space/2), 0.25, space, facecolor=away_primary))

        # Performance color rectangles
        ax.add_patch(Rectangle((0.68-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['H_F'])) if pd.notna(row['H_F']) else 'lightgray'))
        ax.add_patch(Rectangle((0.73-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['Per_H'])) if pd.notna(row['Per_H']) else 'lightgray'))
        ax.add_patch(Rectangle((0.79-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['A_F'])) if pd.notna(row['A_F']) else 'lightgray'))
        ax.add_patch(Rectangle((0.84-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['Per_A'])) if pd.notna(row['Per_A']) else 'lightgray'))
        ax.add_patch(Rectangle((0.9-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_o(row['xG_H'])) if pd.notna(row['xG_H']) else 'lightgray'))
        ax.add_patch(Rectangle((0.95-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_o(row['xG_A'])) if pd.notna(row['xG_A']) else 'lightgray'))
        
        if row['home_score'] > row['away_score']:
            primary = home_primary
            text = home_text
        elif row['home_score'] < row['away_score']:
            primary = away_primary
            text = away_text
        else:
            primary = 'white'
            text = 'black'

        ax.add_patch(Rectangle((0.37-0.005,i_loc - space/2),0.05,space,facecolor=primary))

        # Text annotations
        ax.annotate(str(row['Date'])[:10], (col_x['Date'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['Home'], (col_x['Home'], i_loc), va='center', ha='left', size=7, 
                    color=home_text, fontweight='bold')
        ax.annotate(f"{row['H_F']:.2f}", (col_x['HExp'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['score'], (col_x[''], i_loc), va='center', ha='left', size=7, fontweight='bold',color=text)
        ax.annotate(f"{row['A_F']:.2f}", (col_x['AExp'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['Away'], (col_x['Away'], i_loc), va='center', ha='left', size=7,
                    color=away_text, fontweight='bold')
        ax.annotate(f"{row['Per_H']:.2f}" if pd.notna(row['Per_H']) else '', 
                    (col_x['HPer'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['Per_A']:.2f}" if pd.notna(row['Per_A']) else '', 
                    (col_x['APer'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['xG_H']:.2f}" if pd.notna(row['xG_H']) else '', 
                    (col_x['HxG'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['xG_A']:.2f}" if pd.notna(row['xG_A']) else '', 
                    (col_x['AxG'], i_loc), va='center', ha='left', size=7)

        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.3)
        i_loc -= space
    return fig

def create_schedule_figure(plot_df):
    results = plot_df[plot_df.home_score.isna()].reset_index(drop=True).sort_values('game_date',ascending=True)
    results[' '] = ''
    results['score'] = ''
    results = results[['game_date','home','Pre_Pts_H','score','Pre_Pts_A','away',' ','C_H','C_A',' ','h_win','d_win','a_win']].rename(
        columns={'game_date':'Date','home':'Home','Pre_Pts_H':'H_F','Pre_Pts_A':'A_F','away':'Away','C_H':'HRtg','C_A':'ARtg',
                 'h_exp':'HpG','a_exp':'ApG'})

    fig_height = max(4, len(results) * 0.2)
    fig, ax = plt.subplots(figsize=(7,fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Column x positions
    col_x = {
        'Date':   0.01,
        'Home':   0.12,
        'H':0.37,
        'D':0.42,
        'A':0.48,
        'Away':0.53,
        '1':0.78,
        'HRtg':0.79,
        'ARtg':0.84,
        '2':0.89,
        'HExp':0.90,
        'AExp':0.95
    }

    # Headers
    header_y = (len(results)+0.5)/(len(results)+1)
    for col, x in col_x.items():
        if (col == '1') | (col == '2'):
            pass
        else:
            ha = 'left'
            shift = 0.015 if col in ['H','A','D'] else 0
            ax.annotate(col, (x + shift , header_y), va='center', ha=ha, size=7, weight='bold')

    # Row layout
    top = len(results)/(len(results)+1)
    ax.axhline(top, color='black', linewidth=0.8)
    bottom_margin = 1/(len(results)+1)/10
    total_height = top - bottom_margin
    space = total_height / max(4,len(results))
    i_loc = top - space / 2

    # Vertical dividers (between groups)
    for x in col_x.values():
        ax.vlines(x-0.005, bottom_margin, top, color='black', linewidth=0.5)

    for _, row in results.iterrows():
        home_primary = team_colors[row['Home']]['home_primary']
        home_text = team_colors[row['Home']]['home_secondary']
        away_primary = team_colors[row['Away']]['home_primary']
        away_text = team_colors[row['Away']]['home_secondary']

        # Home team background
        ax.add_patch(Rectangle((0.12-0.005, i_loc - space/2), 0.25, space, facecolor=home_primary))
        # Away team background
        ax.add_patch(Rectangle((0.53-0.005, i_loc - space/2), 0.25, space, facecolor=away_primary))

        # Performance color rectangles
        ax.add_patch(Rectangle((0.37-0.005, i_loc - space/2), 0.05, space,
                    facecolor=cmap(norm_w(row['h_win'])) if pd.notna(row['h_win']) else 'lightgray'))
        ax.add_patch(Rectangle((0.42-0.005, i_loc - space/2), 0.06, space,
                    facecolor=cmap(norm_w(row['d_win'])) if pd.notna(row['d_win']) else 'lightgray'))
        ax.add_patch(Rectangle((0.48-0.005, i_loc - space/2), 0.05, space,
                    facecolor=cmap(norm_w(row['a_win'])) if pd.notna(row['a_win']) else 'lightgray'))

        ax.add_patch(Rectangle((0.79-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(row['HRtg']) if pd.notna(row['HRtg']) else 'lightgray'))
        ax.add_patch(Rectangle((0.84-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(row['ARtg']) if pd.notna(row['ARtg']) else 'lightgray'))
        ax.add_patch(Rectangle((0.9-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['H_F'])) if pd.notna(row['H_F']) else 'lightgray'))
        ax.add_patch(Rectangle((0.95-0.005, i_loc - space/2), 0.05, space,
            facecolor=cmap(norm_p(row['A_F'])) if pd.notna(row['A_F']) else 'lightgray'))

        # Text annotations
        ax.annotate(str(row['Date'])[:10], (col_x['Date'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['Home'], (col_x['Home'], i_loc), va='center', ha='left', size=7, 
                    color=home_text, fontweight='bold')
        ax.annotate(f"{row['h_win']:.0%}", (col_x['H'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['d_win']:.0%}", (col_x['D']+0.005, i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['a_win']:.0%}", (col_x['A'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['H_F']:.2f}", (col_x['HExp'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['A_F']:.2f}", (col_x['AExp'], i_loc), va='center', ha='left', size=7)
        ax.annotate(row['Away'], (col_x['Away'], i_loc), va='center', ha='left', size=7,
                    color=away_text, fontweight='bold')
        ax.annotate(f"{row['HRtg']:.0%}" if pd.notna(row['HRtg']) else '', 
                    (col_x['HRtg'], i_loc), va='center', ha='left', size=7)
        ax.annotate(f"{row['ARtg']:.0%}" if pd.notna(row['ARtg']) else '', 
                    (col_x['ARtg'], i_loc), va='center', ha='left', size=7)

        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.3)
        i_loc -= space
    return fig

def scrollable_plot(fig, height=400):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    
    st.markdown(f"""
        <div style="height:{height}px; overflow-y:scroll; overflow-x:hidden;">
            <img src="data:image/png;base64,{img_base64}" style="width:100%;">
        </div>
    """, unsafe_allow_html=True)

def create_player_mvps(player_stats,matches_df,selected_season,selected_end_date):
    player_stats = player_stats[(player_stats.season == selected_season) & (player_stats.game_date <= selected_end_date)].reset_index(drop=True)
    player_stats.loc[(player_stats.minutesPlayed > 0) & (player_stats.rating.isna()),'rating'] = 6.6
    player_stats['Rtg_Weight'] = player_stats.minutesPlayed * player_stats.rating
    player_stats = player_stats.groupby(['id','name','position']).agg({'team':'unique','minutesPlayed':'sum','Rtg_Weight':'sum'})

    avg = player_stats.Rtg_Weight.sum() / player_stats.minutesPlayed.sum()
    player_stats['Rtg'] = player_stats.Rtg_Weight / player_stats.minutesPlayed
    player_stats['MVPRtg'] = (player_stats.Rtg * player_stats.minutesPlayed + avg * (player_stats.minutesPlayed.max() - player_stats.minutesPlayed)) / player_stats.minutesPlayed.max()
    return player_stats.reset_index()[['name','position','team','MVPRtg']]

def create_mvp_figure(plot_df):
    mvps = plot_df.sort_values('MVPRtg',ascending=False).head(100)

    fig, ax = plt.subplots(figsize=(8,40))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Column x positions
    col_x = {
        '':   0.01,
        'Pos':0.35,
        'Team':0.4,
        'Rtg':   0.9}

    # Headers
    header_y = (len(mvps)+0.5)/(len(mvps)+1)
    for col, x in col_x.items():
        ha = 'left'
        ax.annotate(col, (x , header_y), va='center', ha=ha, size=7, weight='bold')

    # Row layout
    top = len(mvps)/(len(mvps)+1)
    ax.axhline(top, color='black', linewidth=0.8)
    bottom_margin = 1/(len(mvps)+1)/10
    total_height = top - bottom_margin
    space = total_height / max(4,len(mvps))
    i_loc = top - space / 2

    # Vertical dividers (between groups)
    for x in col_x.values():
        ax.vlines(x-0.005, bottom_margin, top, color='black', linewidth=0.5)

    for _, row in mvps.iterrows():
        if len(row['team']) > 1:
            primary = 'white'
            secondary = 'black'
        else:
            primary = team_colors[row['team'][0]]['home_primary']
            secondary = team_colors[row['team'][0]]['home_secondary']

        ax.add_patch(Rectangle((0, i_loc - space/2),1, space, facecolor=primary))
        # Text annotations
        ax.annotate(row['name'], (col_x[''], i_loc), va='center', ha='left', size=7,color = secondary,fontweight='bold')
        ax.annotate(row['position'], (col_x['Pos'], i_loc), va='center', ha='left', size=7,color = secondary)
        ax.annotate(row['team'][0], (col_x['Team'], i_loc), va='center', ha='left', size=7,color = secondary)
        ax.annotate(f"{row['MVPRtg']:.2f}" if pd.notna(row['MVPRtg']) else '', (col_x['Rtg'], i_loc), va='center', ha='left', size=7,color = secondary)
        # Row divider
        ax.axhline(i_loc - space/2, color='black', linewidth=0.3)
        i_loc -= space
    return fig

standings = pd.read_feather('data/standings.ftr')
color_map = pd.read_feather('data/color_map.ftr')
color_map['home_secondary'] = color_map.apply(lambda row: '#FFFFFF' if row['home_primary'] == row['home_secondary'] else row['home_secondary'],axis=1)
team_colors = color_map.to_dict('index')
matches = pd.read_feather('data/matches.ftr')
player_stats = pd.read_feather('data/player_stats.ftr')
team_ratings = pd.read_feather('data/team_ratings.ftr')
team_ratings = team_ratings[['Season','Date']].drop_duplicates().merge(team_ratings[['Season','Team']].drop_duplicates()).merge(
    team_ratings,how='outer').sort_values(['Team','Date'])
team_ratings[['A','B','C']] = team_ratings.groupby(['Season','Team'])[['A','B','C']].ffill()
standings_sims, match_sims = load_standings_sims()

# --- MAIN DASHBOARD ---
tab_standings, tab_team = st.tabs([f"Standings", "Team Profile"])

with tab_standings:
    col1, col2 = st.columns([2,3])
    # --- COLUMN 1: LEFT ---
    with col1:
        subcol1, subcol2, subcol3, subcol4 = st.columns([0.25,1,1,1])
        with subcol1:
            season = sorted(standings_sims['season'].unique(), reverse=True)
            selected_season = st.selectbox("Select Year", options=season, index=0, key="season_picker",label_visibility="collapsed")
        with subcol2:
            dates = sorted(standings_sims[standings_sims['season'] == selected_season]['Sim_Date'].unique(),reverse=True)
            selected_end_date = st.selectbox("Select Date",options=dates,index=0, key="end_date_picker",label_visibility='collapsed')
        with subcol3:
            start_dates = sorted(standings_sims[(standings_sims['season'] == selected_season) & (standings_sims['Sim_Date'] < selected_end_date)]['Sim_Date'].unique(),reverse=True)
            selected_start_date = st.selectbox("Select Relative Date",options=start_dates,index=len(start_dates)-2, key='start_date_picker',label_visibility='collapsed')
        with subcol4:
            options = ['Overall','Eastern','Western']
            selected_type = st.selectbox('Select Conference',options=options,index=0,key='type_picker',label_visibility='collapsed')
        matches_df = create_matches_df(match_sims,matches,team_ratings,selected_season,selected_end_date)
        fig = create_results_figure(matches_df)
        st.markdown("<p style='font-size:14px; font-weight:bold; margin-bottom:2px;'>Results</p>", unsafe_allow_html=True)
        scrollable_plot(fig, height=200)
        fig = create_schedule_figure(matches_df)
        st.markdown("")
        st.markdown("<p style='font-size:14px; font-weight:bold; margin-bottom:2px;'>Schedule</p>", unsafe_allow_html=True)
        scrollable_plot(fig, height=200)
        st.markdown("")
        mvp_df = create_player_mvps(player_stats,matches_df,int(selected_season),selected_end_date)
        st.markdown("<p style='font-size:14px; font-weight:bold; margin-bottom:2px;'>Best Players</p>", unsafe_allow_html=True)
        fig = create_mvp_figure(mvp_df)
        scrollable_plot(fig, height=200)

    with col2:
        standings_df = create_standings_file(standings,standings_sims,team_ratings,selected_season,selected_end_date,selected_start_date).sort_values(['P','GD'],ascending=False)
        if selected_type == 'Eastern':
            standings_df = standings_df[standings_df.conference == 'Eastern']
        elif selected_type == 'Western':
            standings_df = standings_df[standings_df.conference == 'Western']
        else: 
            pass
        fig = plot_standings_table(standings_df.drop(columns='season'))
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        st.markdown(f'<img src="data:image/png;base64,{img_base64}" style="width:100%;">', unsafe_allow_html=True)
        plt.close(fig)

        subcol1, subcol2 = st.columns([0.45,0.55])
        with subcol1:
            fig = plot_ratings_scatter(standings_df.drop(columns='season'), team_colors)
            st.plotly_chart(fig, use_container_width=False)
        with subcol2:
            fig_heatmap = plot_position_heatmap(standings_sims, standings_df, selected_end_date, team_colors)
            st.plotly_chart(fig_heatmap)

with tab_team:
    col1, col2 = st.columns([2,3])
    with col1:
        subcol1, subcol2 = st.columns([1,2])
        with subcol1:
            season = sorted(standings_sims['season'].unique(), reverse=True)
            selected_season = st.selectbox("Select Year", options=season, index=0, key="season_picker2",label_visibility="collapsed")
        with subcol2:
            teams = sorted(standings[standings.season == selected_season].F.unique())
            selected_team = st.selectbox('Select Team',options=teams,key='team_picker',label_visibility='collapsed')
