# OMS Application Architecture and Business Context

Document ID: MR-ARCH-014 | Owner: Application Architecture | Status: Approved 2019

## Business context

The Order Management System (OMS) is the system of record for retail orders
at Meridian Retail. It captures orders from the e-commerce front end and the
store network, allocates inventory, prices and invoices orders, and feeds
downstream fulfilment. Supporting batch processes assist back-office
operations. Peak observed load is approximately 40 orders per minute during
seasonal trading.

## Application overview

The OMS is a monolithic application written in Python 2.7, deployed on a
pair of virtual machines on the on-premises VMware estate. There is no
containerisation. Releases are quarterly, deployed by the operations team
using shell scripts.

## Data architecture

All persistent state resides in an Oracle Database 11g Release 2 instance.
Significant business logic is implemented in approximately 120 PL/SQL
packages, including inventory allocation (OMS_PKG.ALLOCATE_INVENTORY),
pricing, and invoice generation. The application connects via cx_Oracle.

Order documents received from partners are dropped onto an NFS share
mounted at /mnt/nfs/orders. Batch jobs poll this location.

## Integration

- Order intake: SOAP web service exposed to the e-commerce platform.
- Warehouse events: published to IBM MQ (queue manager OMSQM01).
- Invoice archive: nightly upload of generated invoices to object storage.

## Operations

Batch jobs are scheduled with cron on the primary VM. Backups are taken
nightly to the enterprise backup service. A warm disaster recovery copy is
maintained at the secondary data centre. Recovery objectives for this
application are defined in the service catalogue.

## Known constraints

- The Python 2.7 runtime is end of life and no longer receives patches.
- Oracle 11gR2 is on extended support.
- The NFS share is a single point of failure for order intake.
