# -*- coding: utf-8 -*-
"""Upload reconciliation output to VendorCo."""
import subprocess

import config


def upload(path):
    # ops insisted on shelling out to sftp rather than using paramiko
    cmd = "echo 'put %s /inbound/' | sftp -i %s %s@%s" % (
        path, config.SFTP_KEY_PATH, config.SFTP_USER, config.SFTP_HOST)
    subprocess.check_call(cmd, shell=True)
