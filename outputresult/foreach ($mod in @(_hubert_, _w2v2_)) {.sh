foreach ($mod in @("hubert", "w2v2")) { 
    foreach ($el in @("050", "100", "200")) { 
        python mymeasure.py   flists   unit_hyps/$mod/clu$el   unit   --phn_dir trip_tsvs   --lab_sets train-clean-100   --phn_sets train-clean-100   --verbose --output tri_${mod}_$el.npz } }^C

for mod in hubert w2v2; do
    for el in 050 100 200; do
        python \
        /storage/LabJob/Projects/UnitTokenAnalysis`
        `/PurityCalculation/src/mymeasure.py \
         /storage/LabJob/Projects/UnitTokenAnalysis/PurityCalculation/Librispeech_eval/oracle/flists \
          unit_hyps/$mod/clu$el unit --phn_dir trip_tsvs --lab_sets train-clean-100 --phn_sets train-clean-100 --verbose --output tri_${mod}_$el.npz
    done
done
