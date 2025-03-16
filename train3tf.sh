#3TF
data_path=data/encode_cnn_arch
tfs="HaibH1hescSp1Pcr1xUniPk SydhImr90MafkIggrabUniPk SydhK562Znf143IggrabUniPk"
batch_size=192
#batch_size=6752
#"$out_dir"/"$reg"/"$sl"/"$dname"/"$a"
out_dir=saved_models_wrn_cyclic

m_type="wrn"

mkdir -p $out_dir
for i in 1; do
  for w in 1; do
    for d in 10 16 22 28 34 40; do
      for sc in cyclic; do
        for wd in 5e-4; do
          for lr in 1e-1; do
            hparams="--scheduler $sc --w $w --m_type $m_type --opt sgd --lr $lr --d $d --wd $wd --seq_length 101"
            hparamsall=$hparams" --batch_size $batch_size"
            out_diri=$out_dir"/"$m_type"-"$i"/"$d"-"$w"/"
            python train_zeng_pth.py $hparamsall --fname "$data_path/motif_discovery/*/*.npz" --attack attacks.MiddleCrop --save_dir $out_diri/101/motif_discovery/all/MiddleCrop &
            python train_zeng_pth.py $hparamsall --fname "$data_path/motif_occupancy/*/*.npz" --attack attacks.MiddleCrop --save_dir $out_diri/101/motif_occupancy/all/MiddleCrop &
            python train_zeng_pth.py $hparamsall --fname "$data_path/*/*/*.npz" --attack attacks.MiddleCrop --save_dir $out_diri/101/all/all/MiddleCrop &
            wait
            for tf in $tfs; do
              python train_zeng_pth.py $hparams --fname $data_path/motif_discovery/$tf/$tf".npz" --attack attacks.MiddleCrop --save_dir $out_diri/101/motif_discovery/$tf/MiddleCrop &
              python train_zeng_pth.py $hparams --fname $data_path/motif_occupancy/$tf/$tf".npz" --attack attacks.MiddleCrop --save_dir $out_diri/101/motif_occupancy/$tf/MiddleCrop &
            done
            wait
          done
        done
      done
    done
  done
done
#python train_zeng_pth.py --fname data/encode_cnn_arch/motif_discovery/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --attack attacks.MiddleCrop --seq_length 101 --save_dir saved_models/NO/101/motif_discovery/SydhImr90MafkIggrabUniPk/MiddleCrop --reg NO &
#python train_zeng_pth.py --fname data/encode_cnn_arch/motif_discovery/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz --attack attacks.MiddleCrop --seq_length 101 --save_dir saved_models/NO/101/motif_discovery/SydhK562Znf143IggrabUniPk/MiddleCrop --reg NO &
#wait
#python train_zeng_pth.py --fname data/encode_cnn_arch/motif_occupancy/HaibH1hescSp1Pcr1xUniPk/HaibH1hescSp1Pcr1xUniPk.npz --attack attacks.MiddleCrop --seq_length 101 --save_dir saved_models/NO/101/motif_occupancy/HaibH1hescSp1Pcr1xUniPk/MiddleCrop --reg NO &
#python train_zeng_pth.py --fname data/encode_cnn_arch/motif_occupancy/SydhImr90MafkIggrabUniPk/SydhImr90MafkIggrabUniPk.npz --attack attacks.MiddleCrop --seq_length 101 --save_dir saved_models/NO/101/motif_occupancy/SydhImr90MafkIggrabUniPk/MiddleCrop --reg NO &
#python train_zeng_pth.py --fname data/encode_cnn_arch/motif_occupancy/SydhK562Znf143IggrabUniPk/SydhK562Znf143IggrabUniPk.npz --attack attacks.MiddleCrop --seq_length 101 --save_dir saved_models/NO/101/motif_occupancy/SydhK562Znf143IggrabUniPk/MiddleCrop --reg NO &
#wait
read -r line
exit 0
