# feup-machine-learning
Basketball Playoff Qualification


## Tables

Awards -> "playerID","award","year","lgID"
Coaches -> "coachID","year","tmID","lgID","stint","won","lost","post_wins","post_losses"
Players_Teams -> "playerID","year","stint","tmID","lgID","GP","GS","minutes","points","oRebounds","dRebounds","rebounds","assists","steals","blocks","turnovers","PF","fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted","threeMade","dq","PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds","PostdRebounds","PostRebounds","PostAssists","PostSteals","PostBlocks","PostTurnovers","PostPF","PostfgAttempted","PostfgMade","PostftAttempted","PostftMade","PostthreeAttempted","PostthreeMade","PostDQ"
Players -> "bioID","pos","firstseason","lastseason","height","weight","college","collegeOther","birthDate","deathDate"~
Series_post -> "year","round","series","tmIDWinner","lgIDWinner","tmIDLoser","lgIDLoser","W","L"
Teams_post -> "year","tmID","lgID","W","L"
Teams -> "year","lgID","tmID","franchID","confID","divID","rank","playoff","seeded","firstRound","semis","finals","name","o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","tmORB","tmDRB","tmTRB","opptmORB","opptmDRB","opptmTRB","won","lost","GP","homeW","homeL","awayW","awayL","confW","confL","min","attend","arena"


year,lgID,tmID,franchID,confID,divID,rank,playoff,seeded,firstRound,semis,finals,name,o_fgm,o_fga,o_ftm,o_fta,o_3pm,o_3pa,o_oreb,o_dreb,o_reb,o_asts,o_pf,o_stl,o_to,o_blk,o_pts,d_fgm,d_fga,d_ftm,d_fta,d_3pm,d_3pa,d_oreb,d_dreb,d_reb,d_asts,d_pf,d_stl,d_to,d_blk,d_pts,tmORB,tmDRB,tmTRB,opptmORB,opptmDRB,opptmTRB,won_x,lost_x,GP_x,homeW,homeL,awayW,awayL,confW,confL,min,attend,arena,playerID,stint_x,GP_y,GS,minutes,points,oRebounds,dRebounds,rebounds,assists,steals,blocks,turnovers,PF,fgAttempted,fgMade,ftAttempted,ftMade,threeAttempted,threeMade,dq,PostGP,PostGS,PostMinutes,PostPoints,PostoRebounds,PostdRebounds,PostRebounds,PostAssists,PostSteals,PostBlocks,PostTurnovers,PostPF,PostfgAttempted,PostfgMade,PostftAttempted,PostftMade,PostthreeAttempted,PostthreeMade,PostDQ,award,coachID,stint_y,won_y,lost_y,post_wins,post_losses,W,L