#!/bin/sh
# Render the password setting supported by Kvrocks 2.16.0.
set -eu
umask 077

: "${REDIS_PASSWORD:?Set REDIS_PASSWORD before starting Kvrocks.}"

data_dir=/var/lib/kvrocks
config_path="$data_dir/kvrocks.conf"
mkdir -p "$data_dir"

# Kvrocks reads the password from its configuration file.
printf 'bind 0.0.0.0\nport 6379\ndir %s\nrequirepass %s\n' \
  "$data_dir" "$REDIS_PASSWORD" > "$config_path"

exec kvrocks -c "$config_path" --dir "$data_dir"
