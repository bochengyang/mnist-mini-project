#!/usr/bin/env bash
set -Eeo pipefail

# check to see if this file is being run or sourced from another script
_is_sourced() {
	# https://unix.stackexchange.com/a/215279
	[ "${#FUNCNAME[@]}" -ge 2 ] \
		&& [ "${FUNCNAME[0]}" = '_is_sourced' ] \
		&& [ "${FUNCNAME[1]}" = 'source' ]
}

_main() {
  if [ "$1" = 'mnist-mini-project' ]; then
    gunicorn app:app -w 2 --threads 2 -b 0.0.0.0:5000
  else
    exec "$@"
  fi
}

if ! _is_sourced; then
	_main "$@"
fi