import pandas as pd

def getPer(df, pt_df, teams_df):
    """
        Calculate the PER for each player & stores it in df

        Parameters:
            df (DataFrame): The dataframe to store the PER in
            pt_df (DataFrame): The dataframe containing the players_teams data
            teams_df (DataFrame): The dataframe containing the teams data
    """
    pt_df = pd.merge(pt_df, teams_df[['year', 'tmID', 'o_asts', 'o_fgm']], on=['year', 'tmID'], how='left')
    pt_df.rename(columns={'o_asts': 't_asts', 'o_fgm': 't_fgm'}, inplace=True)

    # Get league stats (yearly)
    for index, row in pt_df.iterrows():
        # Assists
        lg_asts = teams_df[(teams_df['year'] == row['year'])]['o_asts'].sum()
        pt_df.at[index, 'lg_asts'] = lg_asts
        # Field Goals Made
        lg_fgm = teams_df[(teams_df['year'] == row['year'])]['o_fgm'].sum()
        pt_df.at[index, 'lg_fgm'] = lg_fgm
        # Field Goals Attempted
        lg_fga = pt_df[(pt_df['year'] == row['year'])]['fgAttempted'].sum()
        pt_df.at[index, 'lg_fgAttempted'] = lg_fga
        # Personal Fouls
        lg_pf = pt_df[(pt_df['year'] == row['year'])]['PF'].sum()
        pt_df.at[index, 'lg_pf'] = lg_pf
        # Free Throws Made
        lg_ftMade = pt_df[(pt_df['year'] == row['year'])]['ftMade'].sum()
        pt_df.at[index, 'lg_ftMade'] = lg_ftMade
        # Free Throws Attempted
        lg_ftAttempted = pt_df[(pt_df['year'] == row['year'])]['ftAttempted'].sum()
        pt_df.at[index, 'lg_ftAttempted'] = lg_ftAttempted
        # Points
        lg_points = pt_df[(pt_df['year'] == row['year'])]['points'].sum()
        # Offesnive Rebounds
        lg_oRebounds = pt_df[(pt_df['year'] == row['year'])]['oRebounds'].sum()
        # Rebounds
        lg_rebounds = pt_df[(pt_df['year'] == row['year'])]['rebounds'].sum()
        # Turnovers
        lg_turnovers = pt_df[(pt_df['year'] == row['year'])]['turnovers'].sum()

        pt_df.at[index, 'factor'] = (2 / 3) - (0.5 * (lg_asts / lg_fgm)) / (2 * (lg_fgm / lg_ftMade))
        pt_df.at[index, 'vop'] = lg_points / (lg_fga - lg_oRebounds + lg_turnovers + 0.44 * lg_ftAttempted)
        pt_df.at[index, 'drb'] = (lg_rebounds - lg_oRebounds) / lg_rebounds


    # Make PER stats for each player
    #df['uPER'] = (1 / pt_df['minutes']) * (
    df['uPER'] = (1 ) * (
        pt_df['threeMade'] + (2/3) * pt_df['assists'] 
        + (2 - pt_df['factor'] * (pt_df['t_asts'] / pt_df['t_fgm'])) * pt_df['fgMade']
        + pt_df['ftMade'] * 0.5 * (1 + (1 - (pt_df['t_asts'] / pt_df['t_fgm'])) + 2/3 * (pt_df['t_asts'] / pt_df['t_fgm']))
        - pt_df['vop'] * pt_df['turnovers'] - pt_df['vop'] * pt_df['drb'] * (pt_df['fgAttempted'] - pt_df['fgMade'])
        - pt_df['vop'] * 0.44 * (0.44 + (0.56 * pt_df['drb'])) * (pt_df['ftAttempted'] - pt_df['ftMade'])
        + pt_df['vop'] * (1 - pt_df['drb']) * (pt_df['rebounds'] - pt_df['oRebounds']) + pt_df['vop'] * pt_df['drb'] * pt_df['oRebounds']
        + pt_df['vop'] * pt_df['steals'] + pt_df['vop'] * pt_df['drb'] * pt_df['blocks']
        - pt_df['PF'] * ((pt_df['lg_ftAttempted'] / pt_df['lg_pf']) - 0.44 * (pt_df['lg_ftAttempted'] / pt_df['lg_pf']) * pt_df['vop'])
    )

    # Standardize PER
    LG_AVG = 15
    df['PER'] = df['uPER'] * (LG_AVG / df['uPER'].mean())
    df.drop(columns=['uPER'], inplace=True)

def getEFF(df, pt_df):
    """
        Calculate the EFF for each player & stores it in df

        Parameters:
            df (DataFrame): The dataframe to store the EFF in
            pt_df (DataFrame): The dataframe containing the players_teams data
    """
    # Make EFF stats for each player
    df['EFF'] = (1 / pt_df['minutes']) * (
    #df['EFF'] = (1) * (
        (pt_df['points']) + pt_df['rebounds'] + pt_df['assists'] + pt_df['steals'] + pt_df['blocks']
        - (pt_df['fgAttempted'] - pt_df['fgMade']) - (pt_df['ftAttempted'] - pt_df['ftMade']) - pt_df['turnovers']
        )