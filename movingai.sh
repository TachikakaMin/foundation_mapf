wget https://movingai.com/benchmarks/mapf/mapf-map.zip
wget https://movingai.com/benchmarks/mapf/mapf-scen-random.zip
wget https://movingai.com/benchmarks/mapf/mapf-scen-even.zip
mkdir -p data_benchmark
unzip mapf-map.zip -d data_benchmark/map_files
unzip mapf-scen-even.zip -d data_benchmark/scens
unzip mapf-scen-random.zip -d data_benchmark/scens

python -m tools.convert_scen_to_bin data_benchmark/scens

 python eval_test.py --model_path model_checkpoint_epoch_5.pth --dataset_paths data_benchmark/input_data/scen-even/den312d-even-1.bin --steps 300