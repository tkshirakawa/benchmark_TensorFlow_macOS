

for /l %%n in (0,1,4) do (
echo [Benchmark %%n]

python Train.py -n "neural_networks/CV_net_Benchmark512.py" ^
                -t "data/dataset_training.csv" ^
                -v "data/dataset_validation.csv" ^
                -o "results" ^
                --update_val_metrics_for_epoch

timeout /nobreak 300
)

