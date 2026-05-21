-- A stored procedure that adds a new correction for a student.
DROP PROCEDURE IF EXISTS AddBonus;
delimiter |
CREATE PROCEDURE AddBonus(user_id int, project_name varchar(255), score int)
  BEGIN
    INSERT INTO projects(name) SELECT project_name
    WHERE NOT EXISTS(SELECT 1 FROM projects WHERE name = project_name);
    SET @project_id = (SELECT id FROM projects WHERE name = project_name);
    INSERT INTO corrections(user_id, project_id, score) VALUES (user_id, @project_id, score); 
  END;
|
delimiter ;
