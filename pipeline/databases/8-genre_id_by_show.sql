-- A script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked.
SELECT t1.title, t2.genre_id FROM tv_shows AS t1, tv_show_genres AS t2 WHERE t1.id = t2.show_id ORDER BY t1.title, t2.genre_id;
