#!/bin/bash
# Pousse le kernel nlp-test-app ; si la limite de 2 sessions GPU est atteinte, réessaie toutes les 3 minutes.
# Usage : bash kaggle/nlp-test-app/pousser_en_boucle.sh   (depuis la racine du dépôt)
set -a; . ./.env; set +a
for i in $(seq 1 120); do
  sortie=$(kaggle kernels push -p kaggle/nlp-test-app 2>&1)
  echo "$(date +%H:%M:%S) $sortie"
  if echo "$sortie" | grep -qi "successfully pushed\|Kernel version"; then exit 0; fi
  if ! echo "$sortie" | grep -qi "Maximum batch GPU session"; then exit 1; fi
  sleep 180
done
exit 2
