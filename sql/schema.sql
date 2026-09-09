-- LoanWise Supabase schema
-- Run this in the Supabase SQL editor (or apply via migration).
-- Requires: Supabase Auth enabled (auth.users), Storage enabled.
--
-- This schema assumes the service-role key is used server-side (bypasses RLS).
-- Postgres RLS policies below are defense-in-depth so that even if the public
-- anon key or a user JWT is used to query the tables directly, an applicant
-- can only see their own applications, and only loan officers / staff can
-- read the queue.

-- ---------------------------------------------------------------------------
-- 1. PROFILES
--    Extends auth.users with a human-readable role.
--    role: 'applicant' | 'loan_officer'
-- ---------------------------------------------------------------------------
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  role text not null default 'applicant' check (role in ('applicant', 'loan_officer')),
  full_name text,
  created_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

create policy "profiles_select_own" on public.profiles
  for select using (auth.uid() = id);

create policy "profiles_select_officer_all" on public.profiles
  for select using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid() and p.role = 'loan_officer'
    )
  );

-- Trigger: auto-create a profile row on signup with default role 'applicant'.
create or replace function public.handle_new_user()
returns trigger
language plpgsql security definer set search_path = public
as $$
begin
  insert into public.profiles (id, role, full_name)
  values (
    new.id,
    coalesce(new.raw_user_meta_data->>'role', 'applicant'),
    coalesce(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name')
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------------------------------------------------------------------------
-- 2. APPLICATIONS
--    One row per loan application. `record` holds the full computed
--    recommendation payload (JSONB) so the existing frontend contract is
--    preserved. `user_id` ties the application to the creating auth user.
-- ---------------------------------------------------------------------------
create table if not exists public.applications (
  application_id text primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  status text not null default 'PENDING_REVIEW'
    check (status in ('PENDING_REVIEW', 'APPROVED', 'REJECTED', 'COUNTER_OFFERED', 'WITHDRAWN')),
  human_review_required boolean not null default false,
  decision_note text,
  decision_by uuid references auth.users(id) on delete set null,
  decided_at timestamptz,
  record jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists applications_user_id_idx on public.applications (user_id);
create index if not exists applications_status_idx on public.applications (status);
create index if not exists applications_review_idx on public.applications (human_review_required);

alter table public.applications enable row level security;

-- Applicants can see only their own applications.
create policy "applications_select_own" on public.applications
  for select using (auth.uid() = user_id);

-- Loan officers can see all applications.
create policy "applications_select_officer_all" on public.applications
  for select using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid() and p.role = 'loan_officer'
    )
  );

-- Applicants can create their own applications.
create policy "applications_insert_own" on public.applications
  for insert with check (auth.uid() = user_id);

-- Loan officers can update any application (for decisions).
create policy "applications_update_officer" on public.applications
  for update using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid() and p.role = 'loan_officer'
    )
  );

-- ---------------------------------------------------------------------------
-- 3. AUDIT_LOG
--    Append-only. Every meaningful state change writes a row.
-- ---------------------------------------------------------------------------
create table if not exists public.audit_log (
  id bigserial primary key,
  application_id text,
  actor uuid references auth.users(id) on delete set null,
  actor_label text not null default 'system',
  action text not null,
  before jsonb,
  after jsonb,
  created_at timestamptz not null default now()
);

create index if not exists audit_log_application_id_idx on public.audit_log (application_id);
create index if not exists audit_log_created_at_idx on public.audit_log (created_at);

alter table public.audit_log enable row level security;

-- Loan officers can read the audit trail.
create policy "audit_select_officer" on public.audit_log
  for select using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid() and p.role = 'loan_officer'
    )
  );

-- No row may be updated or deleted. (No policies for UPDATE/DELETE; supabase
-- still allows them with service role, but this schema exposes no such
-- policy to end users, and the API has no such endpoint.)

-- ---------------------------------------------------------------------------
-- 4. DOCUMENTS
--    Metadata for files stored in Supabase Storage.
--    The raw bytes live in the 'loanwise-documents' bucket (private).
-- ---------------------------------------------------------------------------
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  application_id text not null references public.applications(application_id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  filename text not null,
  content_type text not null,
  size_bytes bigint not null,
  storage_path text not null,
  uploaded_at timestamptz not null default now()
);

create index if not exists documents_application_id_idx on public.documents (application_id);

alter table public.documents enable row level security;

-- Applicants can only list documents on their own applications.
create policy "documents_select_own" on public.documents
  for select using (
    auth.uid() = (select user_id from public.applications a where a.application_id = documents.application_id)
  );

-- Loan officers can list all documents.
create policy "documents_select_officer" on public.documents
  for select using (
    exists (
      select 1 from public.profiles p
      where p.id = auth.uid() and p.role = 'loan_officer'
    )
  );

-- Applicants can upload documents to their own applications.
create policy "documents_insert_own" on public.documents
  for insert with check (
    auth.uid() = (select user_id from public.applications a where a.application_id = documents.application_id)
  );
