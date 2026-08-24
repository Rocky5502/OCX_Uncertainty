# Office and Remote Collection Runbook

## Recruitment Target

- Distribute up to 100 unique invitation codes.
- Target 40 completed sessions and at least 36 analyzable participants.
- Do not treat nonresponding invitees as participants.
- The server closes new enrollment after 40 completed post-study forms;
  already enrolled participants may finish.

## Prerequisite

Real collection remains disabled until the institutional approval or exemption
for protocol `opencoderx_human_review_v2_invitation_pool` is recorded in a
completed ethics file. Do not replace approval fields with invented values.

## Shared Office Computer

1. Give every attendee one unused code from
   `human_study/frozen/invitation_codes.csv`.
2. Start the empirical server with the approved ethics file and a secure data
   directory that is backed up but not committed to Git.
3. Open `/start` in a private browser window.
4. After a participant finishes, select **Next participant**, close the private
   window, open a new private window, and enter the next assigned code.
5. Do not allow two people to use the same code.

```bash
PYTHONPATH=. .venv/bin/python human_study/serve_study.py \
  --mode empirical \
  --ethics-approval-file /secure/path/ethics_approval.json \
  --data-dir /secure/path/opencoderx_human_data \
  --host 0.0.0.0 --port 8765
```

On the same computer, use `http://127.0.0.1:8765/start`. Other computers on
the same trusted office network use `http://OFFICE-IP:8765/start`. Confirm local
institutional network rules before allowing inbound connections.

## Progress Monitor

Run this in a second terminal:

```bash
PYTHONPATH=. .venv/bin/python human_study/study_status.py \
  --data-dir /secure/path/opencoderx_human_data \
  --out-csv /secure/path/opencoderx_human_data/participant_progress.csv \
  --watch-seconds 5
```

The monitor reports only study codes and completion state. It never reports
names, contact details, employers, code contents, or response values.

## Remote Participants

Do not expose the dependency-free development server directly to the public
Internet. A remote campaign requires an institution-managed HTTPS endpoint,
access logging policy, backups, firewall rules, and a privacy review consistent
with the ethics determination. Send each participant the HTTPS start link and
their own code through a private channel.

## Invitation Message

Use the approved consent and recruitment wording. A compact operational suffix
is:

> Open the study link and enter your assigned participant code: `[CODE]`.
> Please do not share the code. The session contains a tutorial and 12 code
> review tasks. On a shared computer, begin only after the previous participant
> has selected “Next participant” and closed their private browser window.
