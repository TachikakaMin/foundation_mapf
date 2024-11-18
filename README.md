# foundation_mapf

# the number of binary bits can be used to represent the number of agents
    args.agent_dim = int(np.ceil(np.log2(args.max_agent_num)))
    # every grid of a input map is represented by a feature vector with feature_dim*2+1 features
    # the first feature represents the existence of obstacle
    # the next feature_dim features represents the goal position of a specific agent
    # the next feature_dim features represents the start position of a specific agent
    feature_channels = args.agent_dim * 2 + 1

## Map and Path Generation

first modify params in map_generation/gen_map_and_path.sh, then
```bash
cd map_generation
bash gen_map_and_path.sh
```