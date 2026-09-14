-- Store Vision AI: core Supabase performance/RLS alignment
-- This migration mirrors the database-side migration already applied to the
-- connected Supabase project. It is intentionally idempotent.

create index if not exists visits_store_id_idx
  on public.visits (store_id);

create index if not exists item_events_visit_id_idx
  on public.item_events (visit_id);

create index if not exists billing_events_visit_id_idx
  on public.billing_events (visit_id);

create index if not exists alerts_visit_id_idx
  on public.alerts (visit_id);

alter table public.stores enable row level security;
alter table public.visits enable row level security;
alter table public.item_events enable row security;
alter table public.billing_events enable row security;
alter table public.alerts enable row security;

drop policy if exists stores_authenticated_access on public.stores;
create policy stores_authenticated_access
  on public.stores
  for all
  to authenticated
  using ((select auth.uid()) is not null)
  with check ((select auth.uid()) is not null);

drop policy if exists visits_authenticated_access on public.visits;
create policy visits_authenticated_access
  on public.visits
  for all
  to authenticated
  using ((select auth.uid()) is not null)
  with check ((select auth.uid()) is not null);

drop policy if exists item_events_authenticated_access on public.item_events;
create policy item_events_authenticated_access
  on public.item_events
  for all
  to authenticated
  using ((select auth.uid()) is not null)
  with check ((select auth.uid()) is not null);

drop policy if exists billing_events_authenticated_access on public.billing_events;
create policy billing_events_authenticated_access
  on public.billing_events
  for all
  to authenticated
  using ((select auth.uid()) is not null)
  with check ((select auth.uid()) is not null);

drop policy if exists alerts_authenticated_access on public.alerts;
create policy alerts_authenticated_access
  on public.alerts
  for all
  to authenticated
  using ((select auth.uid()) is not null)
  with check ((select auth.uid()) is not null);

comment on table public.stores is 'Store Vision AI store records';
comment on table public.visits is 'Store Vision AI customer visit records';
comment on table public.item_events is 'Store Vision AI observed item events';
comment on table public.billing_events is 'Store Vision AI billing/counter events';
comment on table public.alerts is 'Store Vision AI security alert candidates';