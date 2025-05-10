import time
import statistics
import torch
import datetime

# vllm 相关
from vllm import LLM, SamplingParams
# sglang 相关
import sglang as sgl

# 模型路径
model_path = (
    "/fs-computility/ai-shen/puyuan/model/huggingface/hub/models--Qwen--Qwen2.5-0.5B/"
    "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)

# 定义多样化的 prompt 列表
prompts = [
    "Hello, my name is John Doe.",
    "请详细描述一下人工智能在医疗领域的应用前景。",
    "What are the benefits and drawbacks of remote work?",
    "简述一下区块链技术的基本原理。",
    "Explain the process of photosynthesis in simple terms.",
    "请给出一个简单易懂的量子计算介绍。",
    "How does the economy react in times of deflation?",
    "请写一首关于春天的诗，要求语言清新。",
    "Describe the cultural impact of social media on modern society.",
    "请简单介绍一下机器学习和深度学习的区别。"
]

# 采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def count_tokens(text: str) -> int:
    """
    简单的 token 计数函数：
    - 如果文本包含空格，则按空格分词
    - 否则，认为每个字符就是一个 token
    """
    if " " in text:
        return len(text.split())
    return len(text)


def print_header(backend_name, sampling_config):
    """打印测试开始的头部信息"""
    print("=" * 60)
    print(f"开始测试 {backend_name} 后端")
    print("当前时间:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"模型路径: {model_path}")
    print(f"采样参数: {sampling_config}")
    print("=" * 60, "\n")


def test_vllm_backend(prompts, model_path):
    """
    调用 vllm 后端进行文本生成测试，并统计生成时间、每个 token 的耗时和输出详情
    """
    print_header("vllm", sampling_params)
    # 初始化 vllm 后端
    vllm_llm = LLM(model=model_path, task="generate")

    latencies = []
    token_times = []
    outputs_info = []
    total_start = time.perf_counter()
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[vllm] 测试 {idx}/{len(prompts)}:")
        prompt_start_time = time.perf_counter()
        # 调用 generate 接口生成文本
        output = vllm_llm.generate(prompt, sampling_params=sampling_params)
        elapsed = time.perf_counter() - prompt_start_time
        latencies.append(elapsed)
        print(f"output:{output}")
        token_count =  len(output[0].outputs[0].token_ids)
        token_time = elapsed / token_count if token_count > 0 else 0
        token_times.append(token_time)
        outputs_info.append(output[0].outputs[0].text)
        print(f"Prompt: {prompt}")
        print(f"Output (字符长度 {len(output[0].outputs[0].text)}; token 数量 {token_count}):")
        print(output[0].outputs[0].text)
        print(f"本次生成耗时: {elapsed:.4f} 秒")
        print(f"每个 token 平均耗时: {token_time:.4f} 秒")
        print("-" * 60)

    total_elapsed = time.perf_counter() - total_start
    mean_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    throughput = len(prompts) / sum(latencies)
    mean_token_time = statistics.mean(token_times)

    print("\n--- vllm 后端性能统计 ---")
    print(f"总测试用时: {total_elapsed:.4f} 秒")
    print(f"平均响应时间: {mean_latency:.4f} 秒")
    print(f"响应时间标准差: {std_latency:.4f} 秒")
    print(f"吞吐量: {throughput:.2f} 次/秒")
    print(f"每个 token 平均耗时: {mean_token_time:.4f} 秒")
    print("=" * 60, "\n")
    
    # 释放 GPU 上未使用的缓存内存
    torch.cuda.empty_cache()


def test_sglang_backend(prompts, model_path):
    """
    调用 sglang 后端进行文本生成测试，并统计生成时间、每个 token 的耗时和输出详情
    """
    sampling_config = {"temperature": 0.8, "top_p": 0.95}
    print_header("sglang", sampling_config)
    sgl_llm = sgl.Engine(model_path=model_path)

    latencies = []
    token_times = []
    outputs_info = []
    total_start = time.perf_counter()
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[sglang] 测试 {idx}/{len(prompts)}:")
        prompt_start_time = time.perf_counter()
        # sglang.generate 接口接收 prompt 列表，返回的是列表结果
        outputs = sgl_llm.generate([prompt], sampling_params=sampling_config)
        elapsed = time.perf_counter() - prompt_start_time
        latencies.append(elapsed)
        # print(f"outputs:{outputs}")
        # import ipdb;ipdb.set_trace()
        output_text = outputs[0].get('text', '无文本输出')

        token_count =  outputs[0]["meta_info"]['completion_tokens']
        token_time = elapsed / token_count if token_count > 0 else 0
        token_times.append(token_time)
        outputs_info.append(output_text)

        print(f"Prompt: {prompt}")
        print(f"Output (字符长度 {len(output_text)}; token 数量 {token_count}):")
        print(output_text)
        print(f"本次生成耗时: {elapsed:.4f} 秒")
        print(f"每个 token 平均耗时: {token_time:.4f} 秒")
        print("-" * 60)

    total_elapsed = time.perf_counter() - total_start
    mean_latency = statistics.mean(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    throughput = len(prompts) / sum(latencies)
    mean_token_time = statistics.mean(token_times)

    print("\n--- sglang 后端性能统计 ---")
    print(f"总测试用时: {total_elapsed:.4f} 秒")
    print(f"平均响应时间: {mean_latency:.4f} 秒")
    print(f"响应时间标准差: {std_latency:.4f} 秒")
    print(f"吞吐量: {throughput:.2f} 次/秒")
    print(f"每个 token 平均耗时: {mean_token_time:.4f} 秒")
    print("=" * 60, "\n")
    
    # 释放 GPU 上未使用的缓存内存
    torch.cuda.empty_cache()


def main():
    print("\n脚本开始运行...\n")
    overall_start_time = time.perf_counter()

    # 测试 vllm 后端
    # test_vllm_backend(prompts, model_path)

    # 如果需要同时测试 sglang 后端，请取消下面这行代码的注释
    test_sglang_backend(prompts, model_path)

    overall_elapsed = time.perf_counter() - overall_start_time
    print(f"所有测试完成，总用时: {overall_elapsed:.4f} 秒")
    print("脚本运行结束。")


if __name__ == '__main__':
    main()