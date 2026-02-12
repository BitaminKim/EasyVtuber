# 测试环境
### 基于
* Release版本v0.3
* 测试平台为windows10 22H2
* CPU Intel 10400f
* GPU Nvidia 3060 12G, Nvidia 1660 Super 6G, ARC A770 16G，RX580 8G.  

### 测试定量
* Jpeg compression 和 Input Simplify 设为 medium
* 使用debug input 和 debug output
* 不使用eyebrow参数
* 帧数限制60fps  
* 英伟达显卡使用TensorRT测试

观察开始后约5-10秒的数值记录, 即不考虑长期缓存效果(没时间测)  

# 测试结果
请注意测试结果仅限参考显卡大概性能，具体性能取决于用户周边配置和显卡驱动。

GPU | model | VRAM Cache | RAM Cache | Interpolation | SR | Frame Rate | GPU Usage
:---: | :---: | :---: | :---: | :---: | :---: | :---:  | :---:
A770 | sepe_half | N/A | No | No | No | 33fps | 97%
A770 | sepe_full | N/A | No | No | No | 48fps | 97%
A770 | stan_half | N/A | No | No | No | 27fps | 97%
A770 | stan_full | N/A | No | No | No | 29fps | 97%
A770 | sepe_half | N/A | 2GB | No | No | 30fps | 82%
A770 | sepe_full | N/A | 2GB | No | No | 43fps | 78%
A770 | stan_half | N/A | 2GB | No | No | 30fps | 82%
A770 | stan_full | N/A | 2GB | No | No | 23fps | 85%
A770 | sepe_full | N/A | No | No | x2_half | 17fps | 97%
A770 | sepe_full | N/A | No | No | x4_half | 29fps | 97%
A770 | sepe_full | N/A | No | No | x4_full | 25fps | 93%
A770 | sepe_full | N/A | No | x2_half | No | 60fps | 93%
A770 | sepe_full | N/A | No | x3_half | No | 60fps | 85%
A770 | sepe_full | N/A | No | x4_half | No | 60fps | 81%
-- | -- | --| --|--|--|--|--
RX580 | sepe_full | N/A | No | No | No | 12fps | 98%
RX580 | stan_full | N/A | No | No | No | 15fps | 98%
RX580 | sepe_full | N/A | 2GB | No | No | 11fps | 93%
RX580 | stan_full | N/A | 2GB | No | No | 13fps | 93%
RX580 | stan_full | N/A | No | No | x2_half | 10fps | 97%
RX580 | stan_full | N/A | No | No | x2_full | 10fps | 97%
RX580 | stan_full | N/A | No | No | x4_half | 11fps | 98%
RX580 | stan_full | N/A | No | No | x4_full | 11fps | 98%
RX580 | stan_full | N/A | No | x2_half | no | 22fps | 98%
RX580 | stan_full | N/A | No | x2_full | no | 22fps | 98%
RX580 | stan_full | N/A | No | x3_half | no | 28fps | 98%
RX580 | stan_full | N/A | No | x3_full | no | 28fps | 98%
RX580 | stan_full | N/A | No | x4_half | no | 33fps | 98%
RX580 | stan_full | N/A | No | x4_full | no | 33fps | 98%
-- | -- | --| --|--|--|--|--
1660s | sepe_half | No | No | No | No | 38fps | 95%
1660s | sepe_full | No | No | No | No | 28fps | 95%
1660s | stan_half | No | No | No | No | 21fps | 96%
1660s | stan_full | No | No | No | No | 12fps | 98%
1660s | sepe_half | 2GB | No | No | No | 39fps | 95%
1660s | sepe_full | 2GB | No | No | No | 28fps | 95%
1660s | stan_half | 2GB | No | No | No | 29fps | 96%
1660s | stan_full | 2GB | No | No | No | 12fps | 96%
1660s | sepe_half | 2GB | 2GB | No | No | 40fps | 75%
1660s | sepe_full | 2GB | 2GB | No | No | 24fps | 80%
1660s | stan_half | 2GB | 2GB | No | No | 26fps | 83%
1660s | stan_full | 2GB | 2GB | No | No | 12fps | 92%
1660s | sepe_half | No | No | No | x2_half | 23fps | 96%
1660s | sepe_half | No | No | No | x2_full | 18fps | 96%
1660s | sepe_half | No | No | No | x4_half | 23fps | 96%
1660s | sepe_half | No | No | No | x4_full | 26fps | 97%
1660s | sepe_half | No | No | x2_half | No | 63fps | 75%
1660s | sepe_half | No | No | x2_full | No | 63fps | 80%
1660s | sepe_half | No | No | x3_half | No | 63fps | 65%
1660s | sepe_half | No | No | x3_full | No | 64fps | 66%
1660s | sepe_half | No | No | x4_half | No | 64fps | 53%
1660s | sepe_half | No | No | x4_full | No | 64fps | 60%
-- | -- | --| --|--|--|--|--
3060 | sepe_half | No | No | No | No | 63fps | 62%
3060 | sepe_full | No | No | No | No | 59fps | 93%
3060 | stan_half | No | No | No | No | 63fps | 80%
3060 | stan_full | No | No | No | No | 31fps | 96%
3060 | sepe_half | 2GB | No | No | No | 63fps | 60%
3060 | sepe_full | 2GB | No | No | No | 60fps | 93%
3060 | stan_half | 2GB | No | No | No | 63fps | 76%
3060 | stan_full | 2GB | No | No | No | 33fps | 96%
3060 | sepe_half | 2GB | 2GB | No | No | 63fps | 58%
3060 | sepe_full | 2GB | 2GB | No | No | 46fps | 70%
3060 | stan_half | 2GB | 2GB | No | No | 43fps | 62%
3060 | stan_full | 2GB | 2GB | No | No | 27fps | 80%
3060 | sepe_half | No | No | No | x2_half | 53fps | 85%
3060 | sepe_half | No | No | No | x2_full | 30fps | 94%
3060 | sepe_half | No | No | No | x4_half | 49fps | 90%
3060 | sepe_half | No | No | No | x4_full | 45fps | 90%
3060 | sepe_half | No | No | x2_half | No | 64fps | 45%
3060 | sepe_half | No | No | x2_full | No | 63fps | 50%
3060 | sepe_half | No | No | x3_half | No | 64fps | 35%
3060 | sepe_half | No | No | x3_full | No | 64fps | 42%
3060 | sepe_half | No | No | x4_half | No | 64fps | 30%
3060 | sepe_half | No | No | x4_full | No | 64fps | 40%

# 已知问题: 
* A770: rife插帧全精度会出现奇怪结果, waifu2x 超分的全精度会输出全绿色
* RX580 tha3 模型半精度会输出奇怪结果
* 3060 有时出现奇怪运行时占用上升，猜测为插拔显卡导致的驱动问题。

# DirectML 思考
DirectML 允许所有 Windows 显卡设备以及 AIPC 的 NPU 运行此项目，但由于不同设备的 Kernel 在不同精度下实现区别和驱动兼容问题，  
导致 DirectML 的运行结果在某个模型的某些精度出现不可用的扭曲结果，且运行性能与对模型的常识理解不相符。  

例如适用于本项目插帧的RIFE模型的调用封装经历过一轮迭代,   
由原先的n次输入(N=1,C,H,W)独立图像调用RIFE模型生成n个独立结果，  
改为输入(N=n,C,H,W)的叠加图像一次性调用RIFE模型得到叠加结果，  
此改动下模型获得了稍小的显存占用且在TensorRT实现中获得一定性能提升（虽然改动本意并不在此）。  
接着在笔者环境中的因特尔 A770 显卡 DirectML 在RIFE模型迭代后全精度推理结果由原先的符合预期变为扭曲图像，  
半精度迭代后模型不仅保持推理结果符合预期，且相较模型迭代前得到了一定性能提升。

再例如笔者测试某 AMD 核显运行时发现其运行THA3的半精度模型时会因为此设备无法运行其中的某个kernel而将kernel运算单独卸载到cpu进行，  
但保留了其他大部分能在核显上运行的kernel，由此推理模型需要结果在两个设备间反复，带来了极大的性能倒退。

目前阶段，受限于Python OnnxRuntime api 的支持匮乏（相较C++），使用 python 运行 Directml 无法避免上述问题，  
在代码层面无法在运行前做出是否适配的条件判断，也无法在出现kernel卸载到cpu性能剧降时raise错误将运行打断。  

所以，DirectML运行本项目的任意模型出现奇怪结果，或不符合预期的性能倒退，都应该属于意料之中    
换个模型精度，观察调试输出是否有DirectML黄字警告 通常可以解决大部分问题。  
具体某个硬件在DirectML的运行异常修复，已远超本项目的目标范围，请自行验证失效kernel，向硬件厂商和微软求助并期待驱动更新带来解决  
（建议买张N卡吧别折腾自己了）。