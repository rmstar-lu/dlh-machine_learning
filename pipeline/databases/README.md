# Database

## `0-create_database_if_missing.sql`

A script to create database db_0 if it does not yet exist.

## `1-first_table.sql`

A script that creates a table called first_table in the current database.

## `2-list_values.sql`

A script that lists all rows of the table first_table.

## `3-insert_value.sql`

A script that inserts a new row in the table first_table.

## `4-best_score.sql`

A script that lists all records with a score >= 10 in the table second_table.

## `5-average.sql`

A script that computes the score average of all records in the table second_table.

## `6-avg_temperatures.sql`

A script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).

## `7-max_state.sql`

A script that displays the max temperature of each state (ordered by State name).

## `8-genre_id_by_show.sql`

A script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked.

## `9-no_genre.sql`

A script that lists all shows contained in hbtn_0d_tvshows without a genre linked.

## `10-count_shows_by_genre.sql`

A script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.

## `11-rating_shows.sql`

A script that lists all shows from hbtn_0d_tvshows_rate by their rating.

## `12-rating_genres.sql`

A script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.

## `13-uniq_users.sql`

A script that creates a table users with auto increment primary key and a unique column.

## `14-country_users.sql`

A script that creates a table users with auto increment primary key, a unique column and an enum column.

## `15-fans.sql`

A script that ranks country origins of bands, ordered by the number of (non-unique) fans.

## `16-glam_rock.sql`

A script that lists all bands with Glam rock as their main style, ranked by their longevity.

## `17-store.sql`

A script that creates a trigger that decreases the quantity of an item after adding a new order.

## `18-valid_email.sql`

A script that creates a trigger that resets the attribute `valid_email` to 0 only when the email has been changed.

## `19-bonus.sql`

A script that creates a stored procedure to add a new correction for a student.

## `20-average_score.sql`

A script that creates a stored procedure to compute and store the average score for a student.

## `21-div.sql`

A script that creates a function to safely divide two integers.

## `22-list_databases`

A script that lists all databases in MongoDB.

## `23-use_or_create_database`

A script to create or use database my_db.

## `24-insert`

A script to insert one record into the collection school.

## `25-all`

A script to list all documents in the collection school.

## `26-match`

A script to list all documents with a certain name in the collection school.

## `27-count`

A script to get the number of documents in the collection school.

## `28-update`

A script to add a new attribute to update all documents in the collection school matching a certain name.

## `29-delete`

A script to delete all documents match a certain name from the collection school.

## `30-all`

A python function to return all documents from a Mongo collection.

## `31-insert_school.py`

A python function to insert a new document in a Mongo collection based on kwargs.

## `32-update_topics.py`

A python function to change all topics of a school document based on the name.

## `33-schools_by_topic.py`

A python function that returns the list of school having a specific topic.

## `34-log_stats.py`

A python script that provides some stats about nginx logs stored in MongoDB.

## `100-index_my_names.sql`

A script to create an index on the first character of the name column.

## `101-index_name_score.sql`

A script to create an index on the first character of the name column and the score.

