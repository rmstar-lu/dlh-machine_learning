-- A script that lists all shows from hbtn_0d_tvshows_rate by their rating.
SELECT t1.title, SUM(t3.rate) AS rating FROM tv_shows AS t1, tv_show_ratings AS t3 WHERE t1.id = t3.show_id GROUP BY t1.id ORDER BY rating DESC;
