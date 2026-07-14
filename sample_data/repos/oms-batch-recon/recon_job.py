# -*- coding: utf-8 -*-
"""Nightly reconciliation: OMS order totals vs general ledger extract."""
import csv
import datetime
import os

import cx_Oracle

import config
import sftp_feed


def fetch_oms_totals(business_date):
    dsn = cx_Oracle.makedsn(config.ORACLE_HOST, 1521, config.ORACLE_SID)
    conn = cx_Oracle.connect(config.ORACLE_USER, config.ORACLE_PASSWORD, dsn)
    cur = conn.cursor()
    cur.execute(
        "SELECT STORE_ID, SUM(TOTAL_AMOUNT) FROM ORDERS "
        "WHERE TRUNC(ORDER_DATE) = :1 GROUP BY STORE_ID", [business_date])
    return dict(cur.fetchall())


def load_gl_totals(business_date):
    path = os.path.join(config.GL_EXTRACT_DIR, "gl_%s.csv" % business_date.strftime("%Y%m%d"))
    totals = {}
    with open(path) as f:
        for row in csv.reader(f):
            totals[int(row[0])] = float(row[1])
    return totals


def run():
    business_date = datetime.date.today() - datetime.timedelta(days=1)
    oms = fetch_oms_totals(business_date)
    gl = load_gl_totals(business_date)
    report_path = os.path.join(config.REPORT_DIR, "recon_%s.csv" % business_date)
    with open(report_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["store_id", "oms_total", "gl_total", "delta"])
        for store_id in sorted(set(oms) | set(gl)):
            a, b = oms.get(store_id, 0.0), gl.get(store_id, 0.0)
            writer.writerow([store_id, a, b, round(a - b, 2)])
    sftp_feed.upload(report_path)
    print "reconciliation complete: %s" % report_path


if __name__ == "__main__":
    run()
