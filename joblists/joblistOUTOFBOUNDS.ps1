python comparison.py --soft --without  --no_plot --generate --model=CANN --H=2DHO --steps=2000 --dt 0.01 --epochs 1 --layers 2   --init_p 1 0 --init_q 0 0  --folder_name=OUTOFBOUNDS
python simulate.py --generate --steps=1000 --model=CANN --H=2DHO --folder_name=OUTOFBOUNDS --dt=0.01 --init_p 1 0 --init_q 0 0 
python learn.py --method=without --model=CANN --H=2DHO --folder_name=OUTOFBOUNDS --dt=0.01 --epoch=200 
python learn.py --method=soft --model=CANN --H=2DHO --folder_name=OUTOFBOUNDS --dt=0.01 --epoch=200  

python simulate.py --generate --steps=2000 --model=CANN --H=2DHO --folder_name=OUTOFBOUNDS --dt=0.01 --init_p 1 0 --init_q 0 0
python simulate.py --without --steps=2000 --model=CANN --H=2DHO --folder_name=OUTOFBOUNDS --dt=0.01 --init_p 1 0 --init_q 0 0 
python simulate.py --soft --steps=2000 --model=CANN --H=2DHO --folder_name=OUTOFBOUNDS --dt=0.01 --init_p 1 0 --init_q 0 0 


