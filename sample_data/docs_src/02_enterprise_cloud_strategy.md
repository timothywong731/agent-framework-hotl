# Enterprise Cloud Strategy and Patterns

Document ID: MR-STRAT-002 | Owner: Enterprise Architecture | Status: Approved 2023

## Strategic direction

Meridian Retail is cloud-first. Microsoft Azure is the approved strategic
cloud platform for all new and migrated workloads. Exceptions require a
formal waiver from the Architecture Review Board.

## Landing zone

Workloads deploy into the enterprise landing zone: hub-and-spoke network
topology, centralised identity via Entra ID, platform logging, and policy
enforced via Azure Policy. All infrastructure must be defined as code.

## Approved patterns

- Compute: Azure App Service or AKS for containerised workloads.
  Plain IaaS virtual machines require a waiver.
- Data: Azure SQL Managed Instance or Azure Database for PostgreSQL.
  Flexible Server are the approved relational targets.
- Messaging: Azure Service Bus is the strategic messaging platform.
- File transfer: managed SFTP on Azure Blob Storage.
- Batch: Azure Container Apps jobs or Azure Functions timer triggers
  replace VM cron.

## Legacy middleware

IBM MQ is scheduled for retirement (date TBC). No new integrations may be
built on IBM MQ. Migrating applications should plan a path to Azure Service
Bus unless the retirement schedule dictates otherwise.

## Migration approach

Applications are assessed using the 6R model (rehost, replatform,
refactor, repurchase, retire, retain). PaaS-first: replatform is preferred
over rehost where feasible within the migration window.

## FinOps

All resources carry mandatory cost-centre and owner tags. Non-production
environments shut down outside business hours.
