# -*- coding: utf-8 -*-
"""Order file drops arrive on the NFS share."""
import os
import shutil

ORDER_DIR = "/mnt/nfs/orders/incoming"
DONE_DIR = "/mnt/nfs/orders/processed"


def list_order_files():
    return [f for f in os.listdir(ORDER_DIR) if f.endswith(".xml")]


def read_order(filename):
    with open(os.path.join(ORDER_DIR, filename)) as f:
        return f.read()


def mark_done(filename):
    shutil.move(os.path.join(ORDER_DIR, filename), os.path.join(DONE_DIR, filename))
