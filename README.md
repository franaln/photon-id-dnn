Photon ID using DNN
===================

1. Skim/slim the nominal single photon ntuples 

`python do_mini_ntuples.py`

2. Create the input dataframe for training (and validation) from mini ntuples

    * Split train/val samples using event number
    * Remove shower shape outliers
    * Compute weights to match fakes to signal in eta/pt
    * Scale input variables using StandardScaler from sklearn
    * Shuffle events

`python do_train_df.py`
