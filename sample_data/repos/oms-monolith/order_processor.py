# -*- coding: utf-8 -*-
"""Order intake and allocation. Python 2.7 - do not run under Python 3."""
import ConfigParser
import os
import time

import db
import file_store
import s3_uploader

POLL_SECONDS = 30


def process_pending_orders():
    config = ConfigParser.ConfigParser()
    config.read("/etc/oms/oms.ini")
    for filename in file_store.list_order_files():
        print "processing %s" % filename
        order = file_store.read_order(filename)
        order_id = db.insert_order(order)
        db.allocate_inventory(order_id)
        invoice_path = db.generate_invoice(order_id)
        s3_uploader.archive_invoice(invoice_path)
        file_store.mark_done(filename)
        print "order %s complete" % order_id


if __name__ == "__main__":
    while True:
        process_pending_orders()
        time.sleep(POLL_SECONDS)
