# -*- coding: utf-8 -*-
"""Invoice archive. Uploads to AWS S3."""
import boto3

BUCKET = "meridian-oms-invoice-archive"


def archive_invoice(path):
    client = boto3.client("s3", region_name="us-east-1")
    key = "invoices/" + path.split("/")[-1]
    client.upload_file(path, BUCKET, key)
    print "archived %s to s3://%s/%s" % (path, BUCKET, key)
