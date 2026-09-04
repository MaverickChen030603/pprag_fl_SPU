#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-iia100.slis.tsukuba.ac.jp}"
REMOTE_USER="${REMOTE_USER:-${V7_REMOTE_USER:-iiserver31}}"
ROOT="${ROOT:-/home/iiserver31/projects/FedE4RAG-main}"
LOCAL_V7="${LOCAL_V7:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REMOTE_TMP="${REMOTE_TMP:-/tmp/v7_deploy_$(date +%Y%m%d_%H%M%S)}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new -o ConnectTimeout=20}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

remote_ssh() {
  local command="$1"
  if [ -n "${V7_SSH_PASSWORD:-}" ]; then
    V7_EXPECT_PASSWORD="$V7_SSH_PASSWORD" V7_EXPECT_USER="$REMOTE_USER" V7_EXPECT_HOST="$HOST" V7_EXPECT_COMMAND="$command" expect <<'EOF'
set timeout -1
set password $env(V7_EXPECT_PASSWORD)
set user $env(V7_EXPECT_USER)
set host $env(V7_EXPECT_HOST)
set command $env(V7_EXPECT_COMMAND)
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "$user@$host" "$command"
expect {
  -re "(?i)password:" { send "$password\r"; exp_continue }
  eof
}
catch wait result
exit [lindex $result 3]
EOF
  else
    ssh $SSH_OPTS "$REMOTE_USER@$HOST" "$command"
  fi
}

remote_scp_dir() {
  local source="$1"
  local target="$2"
  if [ -n "${V7_SSH_PASSWORD:-}" ]; then
    V7_EXPECT_PASSWORD="$V7_SSH_PASSWORD" V7_EXPECT_USER="$REMOTE_USER" V7_EXPECT_HOST="$HOST" V7_EXPECT_SOURCE="$source" V7_EXPECT_TARGET="$target" expect <<'EOF'
set timeout -1
set password $env(V7_EXPECT_PASSWORD)
set user $env(V7_EXPECT_USER)
set host $env(V7_EXPECT_HOST)
set source $env(V7_EXPECT_SOURCE)
set target $env(V7_EXPECT_TARGET)
spawn scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -r "$source" "$user@$host:$target"
expect {
  -re "(?i)password:" { send "$password\r"; exp_continue }
  eof
}
catch wait result
exit [lindex $result 3]
EOF
  else
    scp -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -r "$source" "$REMOTE_USER@$HOST:$target"
  fi
}

log "Deploying local V7 helper files to $REMOTE_USER@$HOST:$REMOTE_TMP"
remote_ssh "mkdir -p '$REMOTE_TMP'"
remote_scp_dir "$LOCAL_V7" "$REMOTE_TMP/V7_local"

log "Installing scripts into project root and bootstrapping V7 from V6"
remote_ssh "set -euo pipefail
  cd '$ROOT'
  mkdir -p '$ROOT/V7_local_plan'
  cp -a '$REMOTE_TMP/V7_local/.' '$ROOT/V7_local_plan/'
  chmod +x '$ROOT/V7_local_plan/scripts/'*.sh
  ROOT='$ROOT' bash '$ROOT/V7_local_plan/scripts/bootstrap_v7_from_v6.sh'
  cp '$ROOT/V7_local_plan/V7_experiment_set_cn.md' '$ROOT/V7/V7_experiment_set_cn.md'
  cp '$ROOT/V7_local_plan/configs/v7_experiment_matrix.yaml' '$ROOT/V7/v7_experiment_matrix.yaml'
  cp '$ROOT/V7_local_plan/scripts/collect_v7_results.py' '$ROOT/V7/collect_v7_results.py'
  cp '$ROOT/V7_local_plan/scripts/write_v7_analysis.py' '$ROOT/V7/write_v7_analysis.py'
  cp '$ROOT/V7_local_plan/scripts/sync_github_v7.sh' '$ROOT/sync_github_v7.sh'
  chmod +x '$ROOT/run_v7_all.sh' '$ROOT/check_v7_status.sh' '$ROOT/sync_github_v7.sh'
"

log "Running preflight status"
remote_ssh "ROOT='$ROOT' '$ROOT/check_v7_status.sh'"

log "Deployment complete"
log "To start first-pass: ssh $REMOTE_USER@$HOST \"cd '$ROOT' && nohup ./run_v7_all.sh first_pass > v7_nohup.log 2>&1 &\""
