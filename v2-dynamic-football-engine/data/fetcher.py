import pandas as pd
import numpy as np
from statsbombpy import sb

def get_match_passing_matrix(match_id: int, team_name: str):
    """
    Connects to the StatsBomb API, fetches the passing events for a given match and team, and constructs a passing matrix
    Columns represent the player making the pass, rows represent the player receiving the pass. The matrix is normalized to represent probabilities.
    """
    print(f"Fetching passing data for match ID: {match_id}, team: {team_name}...")
    
    # Fetch the match data using StatsBomb API
    all_events = sb.events(match_id=match_id)
    
    # Filter the events to only include successful passes for the specified team
    team_events = all_events[
        (all_events['type'] == 'Pass') & 
        (all_events['team'] == team_name) & 
        (all_events['pass_outcome'].isna())
    ].copy()
    
    # Prevent execution errors if the filtered dataset returns completely blank values
    if team_events.empty:
        return np.zeros((0, 0)), []

    # Isolate the unique raw containing the active player list directly from the match data tracking rows
    active_players = pd.concat([team_events['player'], team_events['pass_recipient']]).dropna().unique()
    
    # Count total outbound passes completed per individual node to determine frequency metrics
    player_pass_counts = team_events['player'].value_counts()
    
    # Isolate the top 11 most active players based on the number of passes made, to focus on the core squad
    players = [player for player in player_pass_counts.index if player in active_players][:11]
    
    # Filter passing networks to only include links among these 11 starting nodes
    team_events = team_events[
        team_events['player'].isin(players) & 
        team_events['pass_recipient'].isin(players)
    ]
    
    # Initialize a blank square Matrix dynamically tracking the length of core active names
    passing_matrix = np.zeros((len(players), len(players)))
    
    # Map player names to matrix indices
    player_to_index = {player: i for i, player in enumerate(players)}
    
    # Populate the passing matrix with counts of successful passes between players
    for _, row in team_events.iterrows():
        passer = row['player']
        recipient = row['pass_recipient']
        
        j = player_to_index[passer] # Column pointer (FROM)
        i = player_to_index[recipient] # Row pointer (TO)
        
        passing_matrix[i, j] += 1.0 # Log the successful pass from passer to recipient
        
    return passing_matrix, players




 

