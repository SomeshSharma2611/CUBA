import sqlite3
conn=sqlite3.connect("vps.db")
cursor=conn.cursor()

query="CREATE TABLE IF NOT EXISTS sys_command(in integer primary key, name VARCHAR(100),path)"