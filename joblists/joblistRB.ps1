 python simulate.py --generate --steps=50000 --model=RB --folder_name=TEST


python learn.py --method=without --model=RB --folder_name=TEST
python learn.py --method=soft --model=RB --folder_name=TEST
python learn.py --method=implicit --model=RB --folder_name=TEST

python simulate.py --steps=500 --generate --folder_name=TEST
python simulate.py --steps=500 --implicit --folder_name=TEST
python simulate.py --steps=500 --soft --folder_name=TEST
python simulate.py --steps=500 --without --folder_name=TEST


python plot_compare.py --plot_m --plot_E --plot_L --folder_name=TEST