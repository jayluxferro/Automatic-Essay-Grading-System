#!/usr/bin/python

import logger as log
import sqlite3

def init():
    conn = sqlite3.connect('db.sqlite')
    conn.row_factory = sqlite3.Row
    return conn

def reset():
    conn = init()
    cursor = conn.cursor()
    cursor.execute("delete from results")
    conn.commit()
    log.warning("[-] Database reset completed..")


def add_data(emb, model, folds, fold, kappa, weight):
    conn = init()
    cursor = conn.cursor()
    log.default("[+] Adding data...")
    cursor.execute("insert into results(emb, model, folds, fold, kappa, weight) values(?, ?, ?, ?, ?, ?)", (emb, model, folds, fold, kappa, weight))
    conn.commit()
    log.success("[+] Data added...")
