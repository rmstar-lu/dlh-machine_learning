-- A script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
SELECT t1.name AS genre, COUNT(t3.id) AS number_of_shows FROM tv_genres AS t1, tv_show_genres AS t2, tv_shows AS t3 WHERE t1.id = t2.genre_id AND t2.show_id = t3.id GROUP BY t2.genre_id ORDER BY number_of_shows DESC;
