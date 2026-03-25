# `doc/` 文档目录

这个目录按源码路径做镜像，文档文件名统一采用 `原文件名 + .md`：

- `models/unet.py` -> `doc/models/unet.py.md`
- `tools/build.sh` -> `doc/tools/build.sh.md`
- `MAPF_online_dataset.py` -> `doc/MAPF_online_dataset.py.md`
- `tools/extensions/lacam_online_native.cpp` -> `doc/tools/extensions/lacam_online_native.cpp.md`
- `tools/profile_online_data.py` -> `doc/tools/profile_online_data.py.md`
- `tools/generate_offline_data.py` -> `doc/tools/generate_offline_data.py.md`
- `scaling_law.py` -> `doc/scaling_law.py.md`
- `gen_online_testset.sh` -> `doc/gen_online_testset.sh.md`

每份文档都尽量覆盖这几类信息：

- 文件在项目里的职责
- 类、函数或脚本入口的作用
- 输入输出、返回值和依赖接口
- 常见调用方式

当前文档覆盖的是仓库内的项目源码和脚本文件，不单独为二进制产物写文档，例如：

- `data_generation_ECBS/ECBS`

这类二进制的调用关系会在相关脚本文档中说明。

在线训练相关入口可优先看：

- `doc/train.py.md`
- `doc/train_args.py.md`
- `doc/MAPF_online_dataset.py.md`
- `doc/tools/extensions/lacam_online_native.cpp.md`
- `doc/tools/profile_online_data.py.md`
- `doc/tools/generate_offline_data.py.md`
- `doc/scaling_law.py.md`
- `doc/gen_online_testset.sh.md`
