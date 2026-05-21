-- A script that creates a trigger that resets the attribute valid_email to 0 only when the email has been changed.
delimiter |
CREATE TRIGGER invalidate_email BEFORE UPDATE ON users
  FOR EACH ROW
  BEGIN
    IF (SELECT email FROM users WHERE id = NEW.id) <> NEW.email THEN
      SET NEW.valid_email = 0;
    END IF;
  END;
|
delimiter ;
