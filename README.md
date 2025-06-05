# Giude

The H2O experiment is based on the modification of llama.cpp, and the branch is main. If you want to use the default branch of llama.cpp, please switch to the "master" branch.The following will introduce the usage method of H2O. For the specific implementation and design, please refer to the paper.

# Collection of heterogeneous device information

The information of heterogeneous devices includes: IO speed, the size of each layer of the model, the delay of model computation for each layer, and the delay of model release for each layer. These pieces of information are used as input for the offline_planning module.

There is an "offline_planning" directory under the "llama.cpp" directory:

```
llama.cpp/offline_planning/model_offline_config/                     /*Store configuration files for different models*/

llama.cpp/offline_planning/offline_planning.py                       /*offline planning multi-armed bandit algorithm*/

llama.cpp/offline_planning/parse_offline_planning_log.py             /*Generate configuration file script*/
 
llama.cpp/offline_planning/read_offline_planning.cpp                 /*Read the output values of the offline planning multi-armed bandit algorithm script*/

llama.cpp/offline_planning/write_config_direct.py                    /*Just write "output values of the offline planning multi-armed bandit algorithm" and the script*/
```



1、Before performing model inference on multiple tokens, we need to collect information from heterogeneous devices and generate a configuration file. First, we need to configure (k, w) using the write_config_direct.py script. Here, k must be 0; otherwise, it is impossible to collect information for each layer. w can be customized and a recommended value is 1. The path is /tmp/shared_offline_planning.bin.

```python
python3 write_config_direct.py -k 0 -w 1 -s /tmp/shared_offline_planning.bin
```

2、Check if the write operation was successful using the./read_offline_planning script:

```shell
g++ read_offline_planning.cpp -o read_offline_planning

./read_offline_planning

Read from shared memory: k = 0, w = 1
```

3、The parameter GGML_OFFLINE_PLANNING_LOG was found in the CMakeLists.txt file.

```
option(GGML_OFFLINE_PLANNING_LOG       "ggml: write offline planning to /tmp/offline_planning_log" OFF) change to->

option(GGML_OFFLINE_PLANNING_LOG       "ggml: write offline planning to /tmp/offline_planning_log" ON)
```

4、Compile llama.cpp

```
cmake -B build && cmake --build build --config Release -j 4
```

5、Run the "llama.cpp" program to collect information from heterogeneous devices. Please note that the -n parameter here must be 1, and the --no-warmup option must be enabled.

```
echo 3 > /proc/sys/vm/drop_caches
./llama-cli -m /root/root-data/01-models/qwen2.5-0.5b-instruct-fp16.gguf -p "I believe the meaning of life is" -n 1 -t 1 --no-warmup
```

qwen2.5-0.5b-instruct-fp16.gguf needs to be replaced with your own model.

At this point, the file **/tmp/offline_planning_log** will be generated. It needs to be parsed using the script parse_offline_planning_log.py to generate the model configuration file.

```
python3 parse_offline_planning_log.py --log_path=/tmp/offline_planning_log --output_path ./model_offline_config/qwen2.5-0.5b-instruct-fp16_config

Generated JSON file at: ./model_offline_config/qwen2.5-0.5b-instruct-fp16_config
```

# offline planning

6、After generating the model configuration, we use the offline planning algorithm to take the model configuration as the input, and through the algorithm, we obtain the optimal (k, w) values of the current model within the current memory budget and on the current device.

```
python3 offline_planning.py -m 400 -s /tmp/shared_offline_planning.bin -c ./model_offline_config/qwen2.5-0.5b-instruct-fp16_config
```

# Online inference

7、modify CMakeLists.txt

```
option(GGML_OFFLINE_PLANNING_LOG       "ggml: write offline planning to /tmp/offline_planning_log" ON) change to->

option(GGML_OFFLINE_PLANNING_LOG       "ggml: write offline planning to /tmp/offline_planning_log" OFF)
```

8、recompile

```
rm -rf build

cmake -B build && cmake --build build --config Release -j 4
```

9、runninng llama.cpp inference

```
./llama-cli -m /root/root-data/01-models/qwen2.5-1.5b-instruct-fp16.gguf -p "I believe the meaning of life is" -n 10 -t 1 --no-warmup
```

If the memory budget changes, you need to re-execute the offline_planning.py script. Offline planning will then synchronize the results in real time to the online inference.
