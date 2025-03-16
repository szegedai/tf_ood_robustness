out_fname=./results_2024-09-04_wrn_cyclic_3tf_test.csv
data_path=data/encode_cnn_arch
set=test
#ls -al */*/101/*/*/MiddleCrop/Acc_train/* | awk -F' ' '{print $5" "$9}' | sort -g | less
for d in `ls -d saved_models_wrn_cyclic/*/*/`; do
  echo $d;
  base_cmd="python eval_trans_attack_pth.py --set $set --seq_length 101 --loss xe --metric bacc,auc,aupr --out_fname $out_fname"
  for m_type in model_best_acc model_best_loss; do
    $base_cmd --data_path "$data_path/*/*/*.npz" --model_path $d/101/all/all/MiddleCrop/$m_type
    $base_cmd --data_path "$data_path/*/*/*.npz" --model_path $d/101/motif_discovery/all/MiddleCrop/$m_type
    $base_cmd --data_path "$data_path/*/*/*.npz" --model_path $d/101/motif_occupancy/all/MiddleCrop/$m_type
    for tf in HaibH1hescSp1Pcr1xUniPk SydhImr90MafkIggrabUniPk SydhK562Znf143IggrabUniPk; do
      $base_cmd --data_path data/encode_cnn_arch/motif_discovery/$tf/$tf".npz" --model_path $d/101/motif_discovery/$tf/MiddleCrop/$m_type
      $base_cmd --data_path data/encode_cnn_arch/motif_occupancy/$tf/$tf".npz" --model_path $d/101/motif_occupancy/$tf/MiddleCrop/$m_type
      $base_cmd --data_path data/encode_cnn_arch/motif_discovery/$tf/$tf".npz" --model_path $d/101/motif_occupancy/$tf/MiddleCrop/$m_type
      $base_cmd --data_path data/encode_cnn_arch/motif_occupancy/$tf/$tf".npz" --model_path $d/101/motif_discovery/$tf/MiddleCrop/$m_type
    done
  done
done | tee $out_fname".log"

read -r line
exit 0
