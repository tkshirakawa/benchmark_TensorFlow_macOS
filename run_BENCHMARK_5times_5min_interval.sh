

for i in {0..4}
do echo [BENCHMARK $i]

python Train.py -n "neural_networks/CV_net_Benchmark512.py" \
                -t "data/dataset_training.csv" \
                -v "data/dataset_validation.csv" \
                -o "results" \
                --update_val_metrics_for_epoch

sleep 300   # wait 5 minutes to cool down
done

