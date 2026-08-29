Our organization has adopted a record-audit policy for its data-management
applications. For each app's primary create/update workflows:

1. When a new domain record is created, display a human-readable record ID with a
   domain prefix and a 3+ digit/character suffix where possible (e.g. CALL-001,
   LEAD-001, PAT-001, TASK-001). Where the app already exposes a domain-prefixed
   ID scheme, keep that scheme.
2. Created or updated records must display a "Last updated" timestamp or date,
   refreshed on update.
3. Create and update actions must show a confirmation message.
4. The created/updated record must be findable from the app's relevant list,
   search, filter, tracking, or status view.

Original CRUD, report, and navigation behavior must not be removed or weakened.
