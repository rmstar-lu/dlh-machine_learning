-- A script that lists all shows contained in hbtn_0d_tvshows without a genre linked.
SELECT t1.title, t2.genre_id FROM tv_shows AS t1 LEFT JOIN tv_show_genres AS t2 ON t1.id = t2.show_id WHERE t2.show_id IS NULL;
