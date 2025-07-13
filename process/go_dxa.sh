OVITOS=/efs-mount/ovito-2.9.0-x86_64/bin/ovitos
SCRIPT=constrictions_analysis.py
$OVITOS $SCRIPT $1 $2
rm $1
