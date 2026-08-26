-- Store Vision AI — Supabase schema
-- Run in Supabase: Project > SQL Editor > New query > paste > Run

create table if not exists stores (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    address text,
    created_at timestamptz not null default now()
);

create table if not exists visits (
    id uuid primary key default gen_random_uuid(),
    store_id uuid references stores(id) on delete cascade,
    track_id integer not null,          -- YOLO/ByteTrack track id for this camera session
    camera_id text default 'default',
    entered_at timestamptz not null default now(),
    exited_at timestamptz,
    status text not null default 'in_store'
        check (status in ('in_store', 'exited_clean', 'exited_flagged')),
    unpaid_item_count integer not null default 0
);

create table if not exists item_events (
    id uuid primary key default gen_random_uuid(),
    visit_id uuid references visits(id) on delete cascade,
    item_label text not null,           -- proxy object class from YOLO, e.g. 'backpack', 'bottle'
    confidence real,
    zone text not null,                 -- 'shelf', 'exit'
    detected_at timestamptz not null default now()
);

create table if not exists billing_events (
    id uuid primary key default gen_random_uuid(),
    visit_id uuid references visits(id) on delete cascade,
    item_label text,
    quantity integer not null default 1,
    source text not null default 'manual' check (source in ('manual', 'pos_simulated')),
    billed_at timestamptz not null default now()
);

create table if not exists alerts (
    id uuid primary key default gen_random_uuid(),
    visit_id uuid references visits(id) on delete cascade,
    store_id uuid references stores(id) on delete cascade,
    alert_type text not null default 'unpaid_item_flag',
    unpaid_item_count integer not null,
    status text not null default 'open'
        check (status in ('open', 'confirmed', 'dismissed')),
    created_at timestamptz not null default now(),
    reviewed_at timestamptz,
    reviewed_by text,
    notes text
);

create index if not exists idx_visits_store on visits(store_id);
create index if not exists idx_item_events_visit on item_events(visit_id);
create index if not exists idx_billing_events_visit on billing_events(visit_id);
create index if not exists idx_alerts_status on alerts(status);

-- Row Level Security — open to any authenticated user for demo purposes.
-- Tighten (e.g. scope to store_id owned by the user) before any real deployment.
alter table stores enable row level security;
alter table visits enable row level security;
alter table item_events enable row level security;
alter table billing_events enable row level security;
alter table alerts enable row level security;

create policy "Authenticated read" on stores for select using (auth.role() = 'authenticated');
create policy "Authenticated read" on visits for select using (auth.role() = 'authenticated');
create policy "Authenticated read" on item_events for select using (auth.role() = 'authenticated');
create policy "Authenticated read" on billing_events for select using (auth.role() = 'authenticated');
create policy "Authenticated read" on alerts for select using (auth.role() = 'authenticated');

create policy "Authenticated write" on stores for insert with check (auth.role() = 'authenticated');
create policy "Authenticated write" on visits for all using (auth.role() = 'authenticated') with check (auth.role() = 'authenticated');
create policy "Authenticated write" on item_events for all using (auth.role() = 'authenticated') with check (auth.role() = 'authenticated');
create policy "Authenticated write" on billing_events for all using (auth.role() = 'authenticated') with check (auth.role() = 'authenticated');
create policy "Authenticated write" on alerts for all using (auth.role() = 'authenticated') with check (auth.role() = 'authenticated');
