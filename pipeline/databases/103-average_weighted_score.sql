-- A stored procedure that computes and stores the weighted average score for a student.
DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;
delimiter |
CREATE PROCEDURE ComputeAverageWeightedScoreForUser(user_id int)
  BEGIN
    UPDATE users SET average_score = (SELECT SUM(c.score * p.weight)/SUM(p.weight)
       FROM corrections AS c, projects AS p WHERE c.user_id = user_id AND p.id = c.project_id)
    WHERE id = user_id;
  END;
|
delimiter ;
