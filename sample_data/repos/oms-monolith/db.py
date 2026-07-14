# -*- coding: utf-8 -*-
"""Oracle access layer. Business logic lives in PL/SQL packages."""
import cx_Oracle

DSN = cx_Oracle.makedsn("dboms01.meridian.local", 1521, "OMSPRD")


def _connect():
    # credentials come from /etc/oms/oms.ini [oracle] section
    return cx_Oracle.connect("oms_app", _password_from_config(), DSN)


def insert_order(order):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO ORDERS (PAYLOAD) VALUES (:1) RETURNING ORDER_ID INTO :2",
                [order, cur.var(cx_Oracle.NUMBER)])
    conn.commit()
    return cur.fetchone()[0]


def allocate_inventory(order_id):
    conn = _connect()
    cur = conn.cursor()
    # 120+ PL/SQL packages like this one carry the core business logic
    cur.callproc("OMS_PKG.ALLOCATE_INVENTORY", [order_id])
    conn.commit()


def generate_invoice(order_id):
    conn = _connect()
    cur = conn.cursor()
    cur.callproc("OMS_INVOICE_PKG.GENERATE", [order_id])
    return "/var/oms/invoices/%s.pdf" % order_id


def _password_from_config():
    import ConfigParser
    c = ConfigParser.ConfigParser()
    c.read("/etc/oms/oms.ini")
    return c.get("oracle", "password")
