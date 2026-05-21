-- A script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.
SELECT t1.name, SUM(t3.rate) AS rating FROM tv_genres AS t1, tv_show_genres AS t2, tv_show_ratings AS t3 WHERE t1.id = t2.genre_id AND t2.show_id = t3.show_id GROUP BY t2.genre_id ORDER BY rating DESC;
