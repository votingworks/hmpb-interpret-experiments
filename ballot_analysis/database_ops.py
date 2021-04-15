import json
import os
import sqlite3


DB_NAME = 'ballots.db'


def get_tables(db_path):
    conn = sqlite3.connect(db_path)
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [v[0] for v in res.fetchall() if v[0] != "sqlite_sequence"]
    conn.close()
    return tables


def get_image_records(db_dir, im_name):
    # Connect to sqlite database
    db_path = os.path.join(db_dir, DB_NAME)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT * FROM sheets;")
    front = True
    names = list(map(lambda x: x[0], c.description))
    c.execute(
        "SELECT front_interpretation_json FROM sheets WHERE front_original_filename=?;",
        (im_name,))
    res = c.fetchall()
    if len(res) == 0:
        front = False
        c.execute(
            "SELECT back_interpretation_json FROM sheets WHERE back_original_filename=?;",
            (im_name,))
        res = c.fetchall()

    interp = json.loads(res[0][0])
    mark_info = []
    if 'markInfo' in interp.keys():
        mark_info = interp['markInfo']['marks']
    conn.close()
    return mark_info


