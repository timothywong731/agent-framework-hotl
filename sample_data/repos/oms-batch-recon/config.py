# -*- coding: utf-8 -*-
"""Connection settings for the reconciliation batch."""

ORACLE_HOST = "dboms01.meridian.local"
ORACLE_SID = "OMSPRD"
ORACLE_USER = "recon_batch"
ORACLE_PASSWORD = "Rec0n#2011!"  # TODO move to vault someday

SFTP_HOST = "sftp.vendorco.example"
SFTP_USER = "meridian"
SFTP_KEY_PATH = "/home/recon/.ssh/id_rsa"

GL_EXTRACT_DIR = "/mnt/nfs/finance/gl_extracts"
REPORT_DIR = "/var/recon/reports"
