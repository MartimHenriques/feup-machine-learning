import pandas as pd


DATA_PATH = 'data'
DATA_TEAMS = 'teams.csv'
DATA_COACHES = 'coaches.csv'
DATA_PLAYERS = 'players.csv'
DATA_AWARDS = 'awards_players.csv'
DATA_PLAYERS_TEAMS = 'players_teams.csv'
DATA_SERIES_POST = 'series_post.csv'
DATA_TEAMS_POST = 'teams_post.csv'
DATA_MERGED = 'merged_data.csv'

def prepareTeamsDf(teams_df):
    """
        Simple preparation of the teams data:
        - Remove unnecessary columns
        - Remove nulls
        - Feature engineering
    """
    # Remove unnecessary info
    teams_df.drop(columns=['franchID', 'lgID', 'divID', 'seeded', 'firstRound', 'semis', 'finals', 'name'], inplace=True)
    # Remove nulls
    teams_df.drop(columns=["tmORB","tmDRB","tmTRB","opptmORB","opptmDRB","opptmTRB", 'attend', 'arena'], inplace=True)
    # Collapse wins & losses into one feature
    teams_df['teamWLRatio'] = teams_df['won'] / teams_df['lost']
    teams_df.drop(columns=['won', 'lost', 'GP', 'homeW', 'homeL', 'awayW', 'awayL', 'confW', 'confL', 'min'], inplace=True)
    # Clean up rebounds
    teams_df.drop(columns=['o_oreb', 'o_dreb', 'd_dreb', 'd_oreb'], inplace=True)

    return teams_df

def preparePlayersTeamsDf(df):
    """
        Simple preparation of the players_teams data:
        - Join regular and post season stats
    """
    merging = ['GP', 'minutes', 'points', 'rebounds', 'oRebounds', 'dRebounds', 'assists', 'steals', 'blocks', 'turnovers',
            'PF', 'fgAttempted', 'fgMade', 'ftAttempted', 'ftMade', 'threeAttempted', 'threeMade', 'dq']
    new_pt_df = pd.DataFrame()

    hasUpper = lambda s: any(x.isupper() for x in s)
    for col in merging:
        postName = 'Post' + (col if hasUpper(col) else col.capitalize())
        if col == 'dq':
            postName = 'PostDQ'
        new_pt_df[col] = df[col] + df[postName]

    for col in ['playerID', 'year', 'tmID']:
        new_pt_df[col] = df[col]
    return new_pt_df

def prepareCoachesDf(coaches_df):
    # Collapse wins & losses into one feature
    coaches_df['coachWLRatio'] = (coaches_df['won'] + coaches_df['post_wins']) / (coaches_df['lost'] + coaches_df['post_losses'])
    # Remove unnecessary info
    coaches_df.drop(columns=['lgID', 'stint', 'won', 'lost', 'post_wins', 'post_losses'], inplace=True)
    return coaches_df