Photon ID using DNN
===================

1. Skim/slim the nominal single photon ntuples 

`python do_mini_ntuples.py`

2. Create the dataframes for training, validation and test) from the mini ntuples

    * Split train/val/test samples using event number 
    * Remove shower shape outliers
    * Compute weights to reweight fakes to match signal events in eta/pt
    * Scale input variables using StandardScaler from sklearn
    * Shuffle events

`python do_dataframes.py`

3. Training

`python train.py --df_train df_train.h5 --df_val df_val.h5 --conf train_conf.json --output_dir OUTPUT`
