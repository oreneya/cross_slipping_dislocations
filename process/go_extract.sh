OVITOS=/efs-mount/ovito-2.9.0-x86_64/bin/ovitos
SCRIPT=extract_segments.py
$OVITOS $SCRIPT $1 $2 $3
rm $1/dump.full_run$2.*.ca.gz
